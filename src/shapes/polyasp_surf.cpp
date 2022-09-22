
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

#include <stdio.h>
#include <unistd.h>

#if defined(MTS_ENABLE_OPTIX)
#include "optix/polyasp_surf.cuh"
#endif

#define NUM_POLY_TERMS 10


NAMESPACE_BEGIN(mitsuba)

    static int dbg = 0;
    static int dbg2 = 0;

    template <typename Float, typename Spectrum>
    class PolyAsphSurf final : public Shape<Float, Spectrum> {
        public:
            MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                            get_children_string, parameters_grad_enabled)
                MTS_IMPORT_TYPES()

                using typename Base::ScalarSize;

                using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;
                using Double3 = Vector<Double, 3>;

            PolyAsphSurf(const Properties &props) : Base(props) {
                /// Are the normals pointing inwards relative to the sphere? default: yes for negative curvature, no for positive curvature
                /// This means that the normals are always pointing in the negative z direction by default, i.e. the inside is towards the right halfspace along the z-axis
                m_flip_normals = props.bool_("flip_normals", false);

                // Flip curvature? Can also be specified through negative radius/curvature, note that these combine like -1*-1 = 1
                m_flip = props.bool_("flip", false);


                // Read parameters for polynomial terms
                for(unsigned int n=0; n < NUM_POLY_TERMS; n++) {
                    // Even-term polynomial
                    if(props.has_property("epoly"+std::to_string(n))) {
                        m_poly_is_even = true;
                        m_poly[n] = props.float_("epoly"+std::to_string(n),  0.0f);
                    }
                    // Regular polynomial
                    else if(props.has_property("poly"+std::to_string(n))) {
                        m_poly[n] = props.float_("poly"+std::to_string(n), 0.0f);
                    }
                    else {
                        m_poly[n] = 0.0f;
                    }
                }

                // Update the to_world transform if center is also provided
                m_to_world = m_to_world * ScalarTransform4f::translate(props.point3f("center", 0.f));

                // h limit is common to both
                m_h_lim = props.float_("limit", 0.0f);

                // First lens object - initial parameters
                m_kappa = props.float_("kappa0", 2.f);
                m_r = props.float_("radius0", 2.f);
                m_p =  1.0f / m_r;

                update();

                find_shape_bounds(m_z_min, m_z_max, false);
                find_shape_bounds(m_z_min_base, m_z_max_base, true);

                // How far into z plane?
                fprintf(stdout, "PolyAsphSurf using flip=%s inv_norm=%s kappa=%.2f radius=%.2f (rho=%f) hlim=%.2f z_min=%.2f z_max=%.2f z_min_base=%.2f z_max_base=%.2f\n",
                        m_flip ? "true" : "false",
                        m_flip_normals ? "true" : "false",
                        (double) m_kappa, (double) m_r, (double) m_p, (double) m_h_lim,
                        (double) m_z_min, (double)m_z_max,
                        (double) m_z_min_base, (double)m_z_max_base);

                fprintf(stdout, "polynomial (%s): ", m_poly_is_even ? "even" : "radial");
                for(auto pe = m_poly.begin(); pe != m_poly.end(); pe++) {
                    fprintf(stdout, "%.3E, ", (double)*pe);
                }
                fprintf(stdout, "\n");

                if( isnan( m_z_min ) || isnan( m_z_max)){
                    fprintf(stdout, "nan error\n");
                    fflush(stdout);
                    while(1){};
                }


                set_children();
            }

            void update() {

                std::cerr << "Update\n";
                // Extract center from to_world matrix (25 iterations for numerical accuracy)
                auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

                if (abs(S[0][1]) > 1e-6f || abs(S[0][2]) > 1e-6f || abs(S[1][0]) > 1e-6f ||
                    abs(S[1][2]) > 1e-6f || abs(S[2][0]) > 1e-6f || abs(S[2][1]) > 1e-6f ||
                    abs(S[0][0]-1.0f) > 1e-6f || abs(S[1][1]-1.0f) > 1e-6f || abs(S[2][2]-1.0f) > 1e-6f)
                    Log(Warn, "'to_world' transform shouldn't contain any scaling or shearing!");

                m_center = T;

                // Reconstruct the to_world transform with no scaling / shear
                m_to_world = transform_compose(S, Q, T);
                m_to_object = m_to_world.inverse();

                m_inv_surface_area = 1.0f;
            }


            ScalarBoundingBox3f bbox() const override
            {
                ScalarBoundingBox3f bbox;

                ScalarFloat slack = 1;

                ScalarFloat h_lim = m_h_lim + slack;
                ScalarFloat z_min = m_z_min - slack;
                ScalarFloat z_max = m_z_max + slack;

                bbox.expand(m_center + ScalarPoint3f(h_lim, h_lim, z_max));
                bbox.expand(m_center + ScalarPoint3f(h_lim, -h_lim, z_max));
                bbox.expand(m_center + ScalarPoint3f(-h_lim, -h_lim, z_max));
                bbox.expand(m_center + ScalarPoint3f(-h_lim, h_lim, z_max));

                bbox.expand(m_center + ScalarPoint3f(h_lim, h_lim, z_min));
                bbox.expand(m_center + ScalarPoint3f(h_lim, -h_lim, z_min));
                bbox.expand(m_center + ScalarPoint3f(-h_lim, -h_lim, z_min));
                bbox.expand(m_center + ScalarPoint3f(-h_lim, h_lim, z_min));

                return bbox;
            }

            ScalarFloat surface_area() const override {
                std::cerr << "surface_area\n";
                return 1.0f;
            }

            // =============================================================
            //! @{ \name Sampling routines
            // =============================================================

            PositionSample3f sample_position(Float time, const Point2f &sample,
                                             Mask active) const override {
                MTS_MASK_ARGUMENT(active);

                std::cout << "sample_position\n";
                std::cerr << "sample_position\n";
                std::cerr << sample << "\n";

                Point3f local = warp::square_to_uniform_sphere(sample);

                PositionSample3f ps;
                ps.p = local + m_center;
                ps.n = local;

                if (m_flip_normals)
                    ps.n = -ps.n;

                ps.time = time;
                ps.delta = false;
                ps.pdf = m_inv_surface_area;

                return ps;
            }

            Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
                MTS_MASK_ARGUMENT(active);
                std::cerr << "pdf_position\n";
                std::cout << "pdf_position\n";
                return m_inv_surface_area;
            }

            DirectionSample3f sample_direction(const Interaction3f & /*it*/, const Point2f & /*sample*/,
                                               Mask active) const override
            {
                MTS_MASK_ARGUMENT(active);
                DirectionSample3f result = zero<DirectionSample3f>();

                std::cerr << "sample_direction\n";
                std::cout << "sample_direction\n";

                return result;
            }

            Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                                Mask active) const override {
                MTS_MASK_ARGUMENT(active);
                std::cerr << "pdf_direction\n";
                std::cout << "pdf_direction\n";

                // Sine of the angle of the cone containing the sphere as seen from 'it.p'.
                Float sin_alpha = rcp(norm(m_center - it.p)),
                      cos_alpha = enoki::safe_sqrt(1.f - sin_alpha * sin_alpha);

                return select(sin_alpha < math::OneMinusEpsilon<Float>,
                              // Reference point lies outside the sphere
                              warp::square_to_uniform_cone_pdf(zero<Vector3f>(), cos_alpha),
                              m_inv_surface_area * sqr(ds.dist) / abs_dot(ds.d, ds.n)
                             );
            }

            Mask find_intersections( Double &near_t_, Double &far_t_,
                                     Double3 center,
                                     scalar_t<Double> p, scalar_t<Double> k,
                                     const Ray3f &ray) const
            {

                // Unit vector
                Double3 d(ray.d);

                // Origin
                Double3 o(ray.o);

                // Center of sphere
                Double3 c = Double3(center);

                Double dx = d[0], dy = d[1], dz = d[2];
                Double ox = o[0], oy = o[1], oz = o[2];

                Double x0 = c[0], y0 = c[1], z0 = c[2];

                Double g = -1 * ( 1 + k );

                Double A = -1 * g * sqr(dz) + sqr(dx) + sqr(dy);
                Double B = -1 * g * 2 * oz * dz + 2 * g * z0 * dz + 2 * ox * dx - 2 * x0 * dx + 2 * oy * dy - 2 * y0 * dy - 2 * dz / p;
                Double C = -1 * g * sqr(oz) + g * 2 * z0 * oz - g * sqr(-1*z0) + sqr(ox) - 2 * x0 * ox + sqr(-1*x0) + sqr(oy) - 2 * y0 * oy + sqr(-1*y0) - 2 * oz / p - 2 * -1*z0 / p;

                auto [solution_found, near_t, far_t] = math::solve_quadratic(A, B, C);

                near_t_ = near_t;
                far_t_  = far_t;

                return solution_found;
            }


            void find_shape_bounds(ScalarFloat &z_min, ScalarFloat &z_max, bool base_shape) const
            {
                z_min = math::Infinity<Double>;
                z_max = -math::Infinity<Double>;

                ScalarFloat r_delta = m_h_lim/1000;
                for(ScalarFloat r = 0; r <= (ScalarFloat)m_h_lim; r += r_delta) {
                    ScalarFloat s = base_shape ? conic_sag(r) : aspheric_sag(r);

                    z_min = s < z_min ? s : z_min;
                    z_max = s > z_max ? s : z_max;
                }
            }

            template <typename F>
            F conic_sag(F r) const
            {
                ScalarFloat c = curvature();
                return (sqr(r) * c) / (1 + sqrt(1 - (1 + m_kappa) * sqr(r*c)));
            }

            Double3 conic_normal_vector(Double x, Double y) const
            {
                ScalarFloat c = curvature();
                Double dx = (x * c) / sqrt(1 - (1+m_kappa) * (sqr(x) * sqr(c)));
                Double dy = (y * c) / sqrt(1 - (1+m_kappa) * (sqr(y) * sqr(c)));
                Double dz = -1.0;

                return Double3(dx, dy, dz);
            }

            template <typename F>
            F aspheric_sag(F r) const
            {
                return conic_sag(r) + aspheric_polyterms(r);
            }

            template <typename F>
            F aspheric_polyterms(F r) const
            {
                if(m_poly_is_even) {
                    return aspheric_even_polyterms(r);
                }
                else {
                    return aspheric_radial_polyterms(r);
                }
            }

            template <typename F>
            F aspheric_radial_polyterms(F r) const
            {
                F dz = 0;
                F ri = r;
                for(size_t i=0; i < NUM_POLY_TERMS; i++) {
                    dz += m_poly[i]*ri;
                    ri *= r;
                }
                return dz;
            }

            template <typename F>
            F aspheric_even_polyterms(F r) const
            {
                F dz = 0;
                F ri = r*r;
                for(size_t i=0; i < NUM_POLY_TERMS; i++) {
                    dz += m_poly[i]*ri;
                    ri *= r*r;
                }
                return dz;
            }

            Double3 aspheric_polyterms_derivatives(Double x, Double y) const
            {
                if(m_poly_is_even) {
                    return aspheric_even_polyterms_derivatives(x,y);
                }
                else {
                    return aspheric_radial_polyterms_derivatives(x,y);
                }
            }

            Double3 aspheric_radial_polyterms_derivatives(Double x, Double y) const
            {
                Double r = sqrt( sqr(x) + sqr(y));

                Double dr = 0;
                Double ri = 1/r; // starting at 1/r because we scale with x, y later
                for(size_t i=0; i < NUM_POLY_TERMS; i++) {
                    dr += (i+1)*m_poly[i]*ri;
                    ri *= r;
                }

                return Double3(dr*x, dr*y, 0);
            }

            Double3 aspheric_even_polyterms_derivatives(Double x, Double y) const
            {
                Double r = sqrt( sqr(x) + sqr(y));

                Double dr = 0;
                Double ri = 1; // starting at r/r because we scale with x, y later
                for(size_t i=0; i < NUM_POLY_TERMS; i++) {
                    dr += 2*(i+1)*m_poly[i]*ri;
                    ri *= r*r;
                }

                return Double3(dr*x, dr*y, 0);
            }

            Double aspheric_implicit_fun(Double3 point,
                                         Double3 center) const
            {
                Double x = point[0] - center[0], y = point[1] - center[1], z = point[2] - center[2];
                Double r = sqrt( sqr(x) + sqr(y));

                Double sag = conic_sag(r) + aspheric_polyterms(r);
                return sag - z;
            }

            Double3 aspheric_normal_vector(Double3 point,
                                          Double3 center) const
            {
                Double x = point[0] - center[0], y = point[1] - center[1];

                return conic_normal_vector(x, y) + aspheric_polyterms_derivatives(x, y);
            }

            Mask point_within_surf_bounds( Double3 point,
                                           Double3 center,
                                           scalar_t<Double> z_min,
                                           scalar_t<Double> z_max) const
            {
                Double3 p = point - center;

                Double h = sqrt( sqr(p[0]) + sqr(p[1]) );
                Double z = p[2];

                return (h <= m_h_lim) && (z >= z_min) && (z <= z_max);
            }

            scalar_t<Double> curvature() const
            {
                return (scalar_t<Double>)m_p * (scalar_t<Double>)(m_flip ? -1. : 1.);
            }

            //! @}
            // =============================================================

            // =============================================================
            //! @{ \name Ray tracing routines
            // =============================================================

            PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray,
                                                                Mask active) const override
            {
                MTS_MASK_ARGUMENT(active);

                Double mint = Double(ray.mint);
                Double maxt = Double(ray.maxt);

                // Point-solutions for each sphere
                Double near_t0, far_t0;

                near_t0 = 0.0;
                far_t0  = 0.0;

                scalar_t<Double> p = curvature();
                Mask intersect = find_intersections( near_t0, far_t0,
                                        m_center,
                                        p, (scalar_t<Double>) m_kappa,
                                        ray);

                // Is any hit on the sphere surface which is limited by lens height & depth?
                Mask valid_near = point_within_surf_bounds( ray(near_t0),
                                                     m_center,
                                                     (scalar_t<Double>) m_z_min_base,
                                                     (scalar_t<Double>) m_z_max_base );

                //valid_near = valid_near && (near_t0 >= mint) && (near_t0 < maxt); // Allow negative t at this intermediate stage


                Mask valid_far = point_within_surf_bounds( ray(far_t0),
                                                     m_center,
                                                     (scalar_t<Double>) m_z_min_base,
                                                     (scalar_t<Double>) m_z_max_base );


                //valid_far = valid_far && (far_t0 >= mint) && (far_t0 < maxt); // Allow negative t at this intermediate stage

                Mask valid_base =  intersect && (valid_far || valid_near);

                Double t = select(valid_near, near_t0, far_t0);
                t = select(valid_base, t, mint);
                Double3 P = ray(t);
                Double e = aspheric_implicit_fun(P, m_center);
                Double3 n = aspheric_normal_vector(P, m_center);

                Double ae_min = abs(e);
                Double t_min = t;

                t = 0;                    // Start on zero to follow Spencer & Murty
                Double tolerance = 5e-3;
                scalar_t<Double> exit_tolerance = 2*5e-3;
                unsigned int iter = 0;
                while( any(abs(e) > tolerance) && iter < 16) {
                    Double3 n = aspheric_normal_vector(P, m_center);
                    Double t_delta = - e / dot(ray.d, n);

                    t += t_delta;
                    P = ray(t);
                    e = aspheric_implicit_fun(P, m_center);

                    Mask sel = (abs(e) < ae_min) && (t >= mint) && (t < maxt);

                    ae_min = select(sel, abs(e), ae_min);
                    t_min = select(sel, t, t_min);

                    iter++;
                }

                //fprintf(stdout, "iter: %d, intersect: %d, valid base: %d, valid near: %d, valid far: %d, m_h_lim: %f\n", iter, any(intersect), any(valid_base), any(valid_near), any(valid_far), m_h_lim);
                //std::cout << " e:" << e << " t:" << t << " t_min:" << t_min << " ae_min:" << ae_min << " near:" << ray(near_t0) - m_center << " far:" << ray(far_t0) - m_center << " mint:" << mint << " maxt:" << maxt << " near_t:" << near_t0 << " far_t:" << far_t0 << std::endl;
                //std::cout << " m_z_min_base:" << m_z_min_base << " m_z_max_base:" << m_z_max_base << " m_center:" << m_center << std::endl;

                Mask valid_adj = point_within_surf_bounds( ray(t_min),
                                                       m_center,
                                                       (scalar_t<Double>) m_z_min - exit_tolerance,
                                                       (scalar_t<Double>) m_z_max + exit_tolerance) && (ae_min <= exit_tolerance);

                //std::cout << "valid_adj: " << valid_adj << " t_min:" << t_min << " ae_min:" << ae_min << " near:" << ray(near_t0) - m_center << " far:" << ray(far_t0) - m_center << std::endl;


                Mask valid = valid_adj && (t_min >= mint) && (t_min < maxt);
                /*
                 * Build the resulting ray.
                 * */
                PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();

                pi.t = select( valid, t_min, (scalar_t<Double>)math::Infinity<Float> );

                // Remember to set active mask
                active &= valid;

#if 0
                Ray3f out_ray;
                out_ray.o = ray( pi.t );

                if( p < 0 ){

                    if( 0 || ( ++dbg > 100000 ) ){

                        if( any(  valid ) ) {

                            std::cerr << "point1," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                            std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        }
                        else{
                            //std::cerr << "point2," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                            //std::cerr << "vec2," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        }
                        dbg = 0;
                    }

                    //usleep(1000);
                }
                else{ // !m_flip

                    if( 0 || ( ++dbg2 > 100000 ) ){
                        if( any( valid ) ) {

                            std::cerr << "point1," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                            std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        }
                        else{
                            //std::cerr << "point1," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                            //std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        }
                        dbg2 = 0;
                    }

                    //usleep(1000);
                }
#endif

                pi.shape = this;

                return pi;
            }

            Mask ray_test(const Ray3f &ray, Mask active) const override
            {
                MTS_MASK_ARGUMENT(active);

                Mask solution_found;

                PreliminaryIntersection3f pi;

                // Potentially gets cleared in ray_intersect_preliminary.
                Mask active_ = active;

                pi = ray_intersect_preliminary(ray, active_);

                solution_found = (pi.t != math::Infinity<Float>);

                return solution_found && active;
            }

            SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                             PreliminaryIntersection3f pi,
                                                             HitComputeFlags flags,
                                                             Mask active) const override
            {
                MTS_MASK_ARGUMENT(active);

                // 0xe based on quick look; check interaction.h
                //fprintf(stdout, "HitComputeFlags 0x%x\n", flags);

                bool differentiable = false;
                if constexpr (is_diff_array_v<Float>)
                    differentiable = requires_gradient(ray.o) ||
                        requires_gradient(ray.d) ||
                        parameters_grad_enabled();

                // Recompute ray intersection to get differentiable prim_uv and t
                if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable)){
                    pi = ray_intersect_preliminary(ray, active);
                }

                active &= pi.is_valid();

                SurfaceInteraction3f si = zero<SurfaceInteraction3f>();

                si.t = select(active, pi.t, math::Infinity<Float>);

                Double3 nv = aspheric_normal_vector(ray(pi.t), m_center);

                if( ! m_flip_normals )
                    si.sh_frame.n = normalize( nv );
                else
                    si.sh_frame.n = normalize( -nv );

                si.p = ray(pi.t);

#if 0
                if( 0 || ( ++dbg > 100000 ) ){
                    if(p < 0){
                        std::cerr << "point3," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "\n";
                        std::cerr << "vec3," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        std::cerr << "vec4," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "," << si.sh_frame.n[0] << "," << si.sh_frame.n[1] << "," << si.sh_frame.n[2] << "\n";
                    }
                    else{
                        std::cerr << "point1," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "\n";
                        std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        std::cerr << "vec4," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "," << si.sh_frame.n[0] << "," << si.sh_frame.n[1] << "," << si.sh_frame.n[2] << "\n";
                    }
                    if( dbg == 1 ) usleep(1000);
                    dbg = 0;
                }
#endif

                if (likely(has_flag(flags, HitComputeFlags::UV))) {

                    Vector3f local = m_to_object.transform_affine(si.p);

                    si.uv = Point2f( local.x() / m_r,
                                     local.y() / m_r );

                    if (likely(has_flag(flags, HitComputeFlags::dPdUV))) {

                        si.dp_du = Vector3f( nv[0], 1.0, 0.0 );
                        si.dp_dv = Vector3f( nv[1], 0.0, 1.0 );
                    }
                }

                si.n = si.sh_frame.n;

                if (has_flag(flags, HitComputeFlags::dNSdUV)) {
                    // Should not happen ATM.
                    std::cout << "dNSdUV\n";
                    Log(Warn, "dNSdUV");
                    Log(Error, "dNSdUV");
#if 0
                    ScalarFloat inv_radius = (m_flip_normals ? -1.f : 1.f) / m_radius;
                    si.dn_du = si.dp_du * inv_radius;
                    si.dn_dv = si.dp_dv * inv_radius;
#endif
                }

                return si;
            }

            //! @}
            // =============================================================

            void traverse(TraversalCallback *callback) override {
                std::cerr << "traverse\n";
                std::cout << "traverse\n";
                Base::traverse(callback);
            }

            void parameters_changed(const std::vector<std::string> &/*keys*/) override {
                std::cerr << "parameters_changed\n";
                std::cout << "parameters_changed\n";
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
                        m_optix_data_ptr = cuda_malloc(sizeof(OptixPolyAsphSurfData));

                    ScalarFloat p = curvature();

                    OptixPolyAsphSurfData data = { bbox(), m_to_world, m_to_object, m_center,
                        m_kappa,
                        p,
                        m_r,
                        m_h_lim,
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        m_poly_is_even,
                        m_z_min, m_z_max,
                        m_z_min_base, m_z_max_base,
                        m_flip_normals };

                    for(size_t pi=0; pi < NUM_POLY_TERMS; pi++)
                    {
                        data.poly_coefs[pi] = m_poly[pi];
                    }

                    cuda_memcpy_to_device(m_optix_data_ptr, &data, sizeof(OptixPolyAsphSurfData));
                }
            }
#endif


            std::string to_string() const override {
                std::ostringstream oss;
                oss << "PolyAsphSurf[" << std::endl
                    << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
                    << "  center = "  << m_center << "," << std::endl
                    << "  surface_area = " << surface_area() << "," << std::endl
                    << "  " << string::indent(get_children_string()) << std::endl
                    << "]";
                return oss.str();
            }

            MTS_DECLARE_CLASS()
        private:

                /// Center in world-space
                ScalarPoint3f m_center;
                /// kappa
                ScalarFloat m_kappa;

                /// polynomial
                Vector<ScalarFloat, NUM_POLY_TERMS> m_poly;
                bool m_poly_is_even = false;

                /// curvature
                ScalarFloat m_p;
                /// radius
                ScalarFloat m_r;

                /// limit of height
                ScalarFloat m_h_lim;

                /// flip curvature? Can also be specified through negative radius/curvature, note that these combine like -1*-1 = 1
                bool m_flip;

                /// range in the z-dimension for the surface
                ScalarFloat m_z_min;
                ScalarFloat m_z_max;

                /// range in the z-dimension for the base shape
                ScalarFloat m_z_min_base;
                ScalarFloat m_z_max_base;

                ScalarFloat m_inv_surface_area;

                bool m_flip_normals;
    };

MTS_IMPLEMENT_CLASS_VARIANT(PolyAsphSurf, Shape)
    MTS_EXPORT_PLUGIN(PolyAsphSurf, "PolyAsphSurf intersection primitive");
NAMESPACE_END(mitsuba)
