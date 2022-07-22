
#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

struct OptixAsphSurfData {
    optix::BoundingBox3f bbox;
    optix::Transform4f to_world;
    optix::Transform4f to_object;
    optix::Vector3f center;

    float k;
    float p;
    float r;
    float h_lim;
    float z_lim;

    bool flip_normals;
};

#ifdef __CUDACC__

bool __device__ point_on_lens_surface( Vector3f point, Vector3f center, float z_lim, float h_lim) {

    Vector3f delta0;
    float hyp0;

    delta0 = point - center;

    hyp0 = sqrt( sqr(delta0[0]) + sqr(delta0[1]) + sqr(delta0[2]) );

    float limit;

    float w = (float) z_lim;

    limit = sqrt( sqr(h_lim) + sqr(w) );

    return (hyp0 <= limit);
}

bool __device__ find_intersections0( float &near_t, float &far_t,
                          Vector3f center,
                          float m_p, float m_k,
                          const Ray3f &ray){

    // Unit vector
    Vector3f d = ray.d;

    // Origin
    Vector3f o = ray.o;

    // Center of sphere
    Vector3f c = center;

    float dx = d[0], dy = d[1], dz = d[2];
    float ox = o[0], oy = o[1], oz = o[2];

    float x0 = c[0], y0 = c[1], z0 = c[2];

    float g = -1 * ( 1 + m_k );

    float A = -1 * g * sqr(dz) + sqr(dx) + sqr(dy);
    float B = -1 * g * 2 * oz * dz + 2 * g * z0 * dz + 2 * ox * dx - 2 * x0 * dx + 2 * oy * dy - 2 * y0 * dy - 2 * dz / m_p;
    float C = -1 * g * sqr(oz) + g * 2 * z0 * oz - g * sqr(-1*z0) + sqr(ox) - 2 * x0 * ox + sqr(-1*x0) + sqr(oy) - 2 * y0 * oy + sqr(-1*y0) - 2 * oz / m_p - 2 * -1*z0 / m_p;

    bool solution_found = solve_quadratic(A, B, C, near_t, far_t);

    return solution_found;
}

extern "C" __global__ void __intersection__asphsurf() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixAsphSurfData *asurf = (OptixAsphSurfData *)sbt_data->data;

    // Ray in instance-space
    Ray3f ray = get_ray();

    float near_t0, far_t0;
    bool solution = find_intersections0( near_t0, far_t0,
                                        asurf->center,
                                        asurf->p, asurf->k,
                                        ray);

    if(!solution) {
        return;
    }

    // Is one or both hits on the sphere surface which is limited by lens height & depth?
    bool valid_near = point_on_lens_surface( ray(near_t0),
                    asurf->center,
                    asurf->z_lim, asurf->h_lim );

    valid_near = valid_near && (near_t0 >= ray.mint && near_t0 < ray.maxt);

    bool valid_far = point_on_lens_surface( ray(far_t0),
                     asurf->center,
                     asurf->z_lim, asurf->h_lim );

    valid_far = valid_far && (far_t0 >= ray.mint && far_t0 < ray.maxt);

    if(!(valid_far || valid_near)) {
        return;
    }

    float golden_t0;
    if(valid_near) {
        golden_t0 = near_t0;
    }
    else {
        golden_t0 = far_t0;
    }

    optixReportIntersection( golden_t0, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE );
}


extern "C" __global__ void __closesthit__asphsurf() {
    unsigned int launch_index = calculate_launch_index();

    if (params.is_ray_test()) {
        params.out_hit[launch_index] = true;
    } else {
        const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
        OptixAsphSurfData *asurf = (OptixAsphSurfData *)sbt_data->data;

        // Ray in instance-space
        Ray3f ray = get_ray();

        // Early return for ray_intersect_preliminary call
        if (params.is_ray_intersect_preliminary()) {
            write_output_pi_params(params, launch_index, sbt_data->shape_ptr, 0, Vector2f(), ray.maxt);
            return;
        }

        /* Compute and store information describing the intersection. This is
           very similar to AsphSurf::compute_surface_interaction() */

        Vector3f p;

        // From cylinder.h
        p = ray( ray.maxt );

        Vector3f point = p - asurf->center;

        /*
         * Now compute the unit vector
         * */
        float fx, fy, fz;

        Vector3f ns;

        fx = ( point[0] * asurf->p ) / sqrt( 1 - (1+asurf->k) * (sqr(point[0]) + sqr(point[1])) * sqr(asurf->p) );
        fy = ( point[1] * asurf->p ) / sqrt( 1 - (1+asurf->k) * (sqr(point[0]) + sqr(point[1])) * sqr(asurf->p) );
        fz = -1.0;

        if( ! asurf->flip_normals )
            ns = normalize( Vector3f( fx, fy, fz ) );
        else
            ns = normalize( -1.f * Vector3f( fx, fy, fz ) );

        Vector3f ng = ns;

        Vector2f uv;
        Vector3f dp_du, dp_dv;
        if (params.has_uv()) {

            Vector3f local = asurf->to_object.transform_point(p);

            uv = Vector2f( local.x() / asurf->r,
                             local.y() / asurf->r );

            if (params.has_dp_duv()) {

                dp_du = Vector3f( fx, 1.0, 0.0 );
                dp_dv = Vector3f( fy, 0.0, 1.0 );

            }
        }

        Vector3f dn_du, dn_dv;

        dn_du = dp_du; // <<
        dn_dv = dp_dv; // Was flipped for negative radius

        // Produce all of this
        write_output_si_params(params, launch_index, sbt_data->shape_ptr,
                               0, p, uv, ns, ng, dp_du, dp_dv, dn_du, dn_dv, ray.maxt);
    }
}
#endif
