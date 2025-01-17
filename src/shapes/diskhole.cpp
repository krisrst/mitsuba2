#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

#if defined(MTS_ENABLE_OPTIX)
    #include "optix/diskhole.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-disk:

Diskhole (:monosp:`diskhole`)
-------------------------------------------------

.. pluginparameters::

 * - flip_normals
   - |bool|
   - Is the disk inverted, i.e. should the normal vectors be flipped? (Default: |false|)
 * - to_world
   - |transform|
   - Specifies a linear object-to-world transformation. Note that non-uniform scales are not
     permitted! (Default: none, i.e. object space = world space)

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_disk.jpg
   :caption: Basic example
.. subfigure:: ../../resources/data/docs/images/render/shape_disk_parameterization.jpg
   :caption: A textured disk with the default parameterization
.. subfigend::
   :label: fig-disk

This shape plugin describes a simple disk intersection primitive. It is
usually preferable over discrete approximations made from triangles.

By default, the disk has unit radius and is located at the origin. Its
surface normal points into the positive Z-direction.
To change the disk scale, rotation, or translation, use the
:monosp:`to_world` parameter.

The following XML snippet instantiates an example of a textured disk shape:

.. code-block:: xml

    <shape type="diskhole">
        <bsdf type="diffuse">
            <texture name="reflectance" type="checkerboard">
                <transform name="to_uv">
                    <scale x="2" y="10" />
                </transform>
            </texture>
        </bsdf>
    </shape>
 */

template <typename Float, typename Spectrum>
class Diskhole final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                    get_children_string, parameters_grad_enabled)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarSize;

    Diskhole(const Properties &props) : Base(props) {
        float normal_sign = props.bool_("flip_normals", false) ? -1.0f : 1.0f;
        auto radius = props.float_("radius", 1.0f);
        auto center = props.point3f("center",ScalarPoint3f(0.f, 0.f, 0.f));
        m_rhole = props.float_("rhole",0.0f)/radius;

        m_to_world = m_to_world * ScalarTransform4f::translate(center) *
                                  ScalarTransform4f::scale(ScalarVector3f(radius, radius, 1.f)) *
                                  ScalarTransform4f::scale(ScalarVector3f(1.f, 1.f, normal_sign));

        update();
        set_children();
    }

    void update() {
         // Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
        auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

        if (abs(S[0][1]) > 1e-6f || abs(S[0][2]) > 1e-6f || abs(S[1][0]) > 1e-6f ||
            abs(S[1][2]) > 1e-6f || abs(S[2][0]) > 1e-6f || abs(S[2][1]) > 1e-6f)
            Log(Warn, "'to_world' transform shouldn't contain any shearing!");

        if (!(abs(S[0][0] - S[1][1]) < 1e-6f))
            Log(Warn, "'to_world' transform shouldn't contain non-uniform scaling along the X and Y axes!");

        m_to_object = m_to_world.inverse();
        m_normal = normalize(m_to_world * ScalarNormal3f(0.f, 0.f, 1.f));;

        m_inv_surface_area = 1.f / surface_area();
   }

    ScalarBoundingBox3f bbox() const override {
        ScalarBoundingBox3f bbox;
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(-1.f, -1.f, 0.f)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(-1.f,  1.f, 0.f)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f( 1.f, -1.f, 0.f)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f( 1.f,  1.f, 0.f)));
        return bbox;
    }

    ScalarFloat surface_area() const override {
        auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

        return S[0][0] * S[0][0] * (math::Pi<ScalarFloat> - math::Pi<ScalarFloat> * m_rhole * m_rhole);
    }

    // =============================================================
    //! @{ \name Sampling routines
    // =============================================================

    PositionSample3f sample_position(Float time, const Point2f &sample,
                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Point2f p = warp::square_to_uniform_disk_concentric(sample);

        PositionSample3f ps;
        ps.p    = m_to_world.transform_affine(Point3f(p.x(), p.y(), 0.f));
        ps.n    = m_normal;
        ps.pdf  = m_inv_surface_area;
        ps.time = time;
        ps.delta = false;

        return ps;
    }

    Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return m_inv_surface_area;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray_,
                                                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Ray3f ray     = m_to_object.transform_affine(ray_);
        Float t       = -ray.o.z() * ray.d_rcp.z();
        Point3f local = ray(t);

        // Is intersection within ray segment and disk?
        auto d2 = local.x()*local.x() + local.y()*local.y();
        active = active && t >= ray.mint
                        && t <= ray.maxt
                        && d2 <= 1.f
                        && d2 > m_rhole*m_rhole;

        PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
        pi.t = select(active, t, math::Infinity<Float>);
        pi.prim_uv = Point2f(local.x(), local.y());
        pi.shape = this;

        return pi;
    }

    Mask ray_test(const Ray3f &ray_, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Ray3f ray     = m_to_object.transform_affine(ray_);
        Float t      = -ray.o.z() * ray.d_rcp.z();
        Point3f local = ray(t);

        // Is intersection within ray segment and rectangle?
        auto d2 = local.x()*local.x() + local.y()*local.y();
        return active && t >= ray.mint
                      && t <= ray.maxt
                      && d2 <= 1.f
                      && d2 >= m_rhole*m_rhole;
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        bool differentiable = false;
        if constexpr (is_diff_array_v<Float>)
            differentiable = requires_gradient(ray.o) ||
                             requires_gradient(ray.d) ||
                             parameters_grad_enabled();

        // Recompute ray intersection to get differentiable prim_uv and t
        if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable))
            pi = ray_intersect_preliminary(ray, active);

        active &= pi.is_valid();

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t = select(active, pi.t, math::Infinity<Float>);

        si.p = ray(pi.t);

        if (likely(has_flag(flags, HitComputeFlags::UV))) {
            Float r = norm(Point2f(pi.prim_uv.x(), pi.prim_uv.y())),
                  u = (r-m_rhole)/(1.0f-m_rhole),
                  inv_r = rcp(r);

            Float dr_du = 1.0f-m_rhole;

            Float v = atan2(pi.prim_uv.y(), pi.prim_uv.x()) * math::InvTwoPi<Float>;
            masked(v, v < 0.f) += 1.f;
            si.uv = Point2f(u, v);

            if (likely(has_flag(flags, HitComputeFlags::dPdUV))) {
                Float cos_phi = select(neq(r, 0.f), pi.prim_uv.x() * inv_r, 1.f),
                      sin_phi = select(neq(r, 0.f), pi.prim_uv.y() * inv_r, 0.f);

                si.dp_du = m_to_world * Vector3f( cos_phi, sin_phi, 0.f) * dr_du;
                si.dp_dv = m_to_world * Vector3f(-sin_phi, cos_phi, 0.f);
            }
        }

        si.n          = m_normal;
        si.sh_frame.n = m_normal;

        si.dn_du = si.dn_dv = zero<Vector3f>();

        return si;
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        update();
        Base::parameters_changed();
#if defined(MTS_ENABLE_OPTIX)
        optix_prepare_geometry();
#endif
    }

#if defined(MTS_ENABLE_OPTIX)
    using Base::m_optix_data_ptr;

    void optix_prepare_geometry() override {
        if constexpr (is_cuda_array_v<Float>) {
            if (!m_optix_data_ptr)
                m_optix_data_ptr = cuda_malloc(sizeof(OptixDiskholeData));

            OptixDiskholeData data = { bbox(), m_to_world, m_to_object, (float)m_rhole };

            cuda_memcpy_to_device(m_optix_data_ptr, &data, sizeof(OptixDiskholeData));
        }
    }
#endif

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Diskhole[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
            << "  rhole = " << string::indent(m_rhole) << "," << std::endl
            << "  normal = " << string::indent(m_normal) << "," << std::endl
            << "  surface_area = " << surface_area() << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarNormal3f m_normal;
    ScalarFloat m_rhole;
    ScalarFloat m_inv_surface_area;
};

MTS_IMPLEMENT_CLASS_VARIANT(Diskhole, Shape)
MTS_EXPORT_PLUGIN(Diskhole, "Diskhole intersection primitive");
NAMESPACE_END(mitsuba)
