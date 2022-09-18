import mitsuba
import pytest
import enoki as ek
from enoki.dynamic import Float32 as Float
import warnings

def test01_create(variant_scalar_rgb):
    from mitsuba.core import xml

    s = xml.load_dict({"type" : "diskhole"})
    assert s is not None
    assert s.primitive_count() == 1
    assert ek.allclose(s.surface_area(), ek.pi)


def test02_bbox(variant_scalar_rgb):
    from mitsuba.core import xml, Vector3f, Transform4f

    # radius, rhole and center are in the disk's local space,
    # which has some consequences for usage

    # Can change the size of the diskhole both by setting a radius
    # and further uniform scaling in the to_world transform
    sc = 2.5
    radius = 3.0
    rhole = 1.0
    for translate in [Vector3f([1.3, -3.0, 5]),
                      Vector3f([-10000, 3.0, 31])]:
        s = xml.load_dict({
            "type" : "diskhole",
            "radius" : radius,
            "rhole": rhole,
            "to_world" : Transform4f.translate(translate) * Transform4f.scale((sc, sc, 1.0))
        })
        b = s.bbox()

        assert ek.allclose(s.surface_area(), (sc*radius)**2 * ek.pi - (sc*rhole)**2 * ek.pi)

        assert b.valid()
        assert ek.allclose(b.center(), translate)
        assert ek.allclose(b.min, translate - [sc*radius, sc*radius, 0.0])
        assert ek.allclose(b.max, translate + [sc*radius, sc*radius, 0.0])

    # Can also set center as a property instead of using to_world/translate
    for translate in [Vector3f([1.3, -3.0, 5]),
                      Vector3f([-10000, 3.0, 31])]:
        s = xml.load_dict({
            "type" : "diskhole",
            "radius" : radius,
            "rhole": rhole,
            "center": translate,
            "to_world" :  Transform4f.scale((1, 1, 1.0))
        })
        b = s.bbox()


        assert b.valid()
        assert ek.allclose(b.center(), translate)
        assert ek.allclose(b.min, translate - [radius, radius, 0.0])
        assert ek.allclose(b.max, translate + [radius, radius, 0.0])


def test03_ray_intersect(variant_scalar_rgb):
    from mitsuba.core import xml, Ray3f, Vector3f, Transform4f
    from mitsuba.render import HitComputeFlags

    for rhole in [0, 0.5]:
        for r in [1, 3, 5]:
            for translate in [Vector3f([0.0, 0.0, 0.0]),
                            Vector3f([1.0, -5.0, 0.0])]:
                s = xml.load_dict({
                    "type" : "scene",
                    "foo" : {
                        "type" : "diskhole",
                        "radius" : r,
                        "rhole": rhole,
                        "to_world" : Transform4f.translate(translate)
                    }
                })

                # grid size
                n = 5
                for x in ek.linspace(Float, -1, 1, n):
                    for y in ek.linspace(Float, -1, 1, n):
                        x = 1.1 * r * (x - translate[0])
                        y = 1.1 * r * (y - translate[1])

                        ray = Ray3f(o=[x, y, -10], d=[0, 0, 1],
                                    time=0.0, wavelengths=[])
                        si_found = s.ray_test(ray)

                        assert si_found == ((x**2 + y**2 <= r*r) and (x**2 + y**2 >= rhole*rhole))

                        if si_found:
                            ray = Ray3f(o=[x, y, -10], d=[0, 0, 1],
                                        time=0.0, wavelengths=[])

                            si = s.ray_intersect(ray, HitComputeFlags.All | HitComputeFlags.dNSdUV)
                            ray_u = Ray3f(ray)
                            ray_v = Ray3f(ray)
                            eps = 1e-4
                            ray_u.o += si.dp_du * eps
                            ray_v.o += si.dp_dv * eps
                            si_u = s.ray_intersect(ray_u)
                            si_v = s.ray_intersect(ray_v)

                            if si_u.is_valid():
                                dp_du = (si_u.p - si.p) / eps
                                dn_du = (si_u.n - si.n) / eps
                                assert ek.allclose(dp_du, si.dp_du, atol=2e-3)
                                assert ek.allclose(dn_du, si.dn_du, atol=2e-6)
                            if si_v.is_valid():
                                dp_dv = (si_v.p - si.p) / eps
                                dn_dv = (si_v.n - si.n) / eps
                                assert ek.allclose(dp_dv, si.dp_dv, atol=2e-3)
                                assert ek.allclose(dn_dv, si.dn_dv, atol=2e-6)


def test03b_ray_intersect(variant_gpu_rgb):
    from mitsuba.core import xml, Ray3f, Vector3f, Transform4f
    from mitsuba.render import HitComputeFlags

    for rhole in [0, 0.5]:
        for r in [1, 3, 5]:

            shape = xml.load_dict({
                    "type" : "diskhole",
                    "radius" : r,
                    "rhole": rhole,
            })

            # grid size
            n = 5
            for x in ek.linspace(Float, -1, 1, n):
                for y in ek.linspace(Float, -1, 1, n):
                    x = 1.1 * r * x
                    y = 1.1 * r * y

                    ray = Ray3f(Vector3f(x, y, -10.0), Vector3f(0.0, 0.0, 1.0), 0, [])
                    si_found = shape.ray_test(ray)
                    assert si_found == ((x**2 + y**2 <= r*r) and (x**2 + y**2 >= rhole*rhole))

                    if si_found == True:
                        ray = Ray3f(o=[x, y, -10], d=[0, 0, 1],
                                    time=0.0, wavelengths=[])

                        si = shape.ray_intersect(ray, HitComputeFlags.All | HitComputeFlags.dNSdUV)
                        ray_u = Ray3f(ray)
                        ray_v = Ray3f(ray)
                        eps = 1e-4
                        ray_u.o += si.dp_du * eps
                        ray_v.o += si.dp_dv * eps
                        si_u = shape.ray_intersect(ray_u)
                        si_v = shape.ray_intersect(ray_v)

                        if si.is_valid() == False:
                            warnings.warn("ray_intersect failed for successful ray_test at (x,y) = ({}, {})".format(x,y))
                            continue

                        if si_u.is_valid() == True:
                            dp_du = (si_u.p - si.p) / eps
                            dn_du = (si_u.n - si.n) / eps
                            assert ek.allclose(dp_du, si.dp_du, atol=2e-3)
                            assert ek.allclose(dn_du, si.dn_du, atol=2e-6)
                        if si_v.is_valid() == True:
                            dp_dv = (si_v.p - si.p) / eps
                            dn_dv = (si_v.n - si.n) / eps
                            assert ek.allclose(dp_dv, si.dp_dv, atol=2e-3)
                            assert ek.allclose(dn_dv, si.dn_dv, atol=2e-6)

def test04_differentiable_surface_interaction_ray_forward(variant_gpu_autodiff_rgb):
    from mitsuba.core import xml, Ray3f, Vector3f, UInt32

    shape = xml.load_dict({'type' : 'diskhole'})

    ray = Ray3f(Vector3f(0.1, -0.2, -10.0), Vector3f(0.0, 0.0, 1.0), 0, [])
    pi = shape.ray_intersect_preliminary(ray)

    ek.set_requires_gradient(ray.o)
    ek.set_requires_gradient(ray.d)

    # If the ray origin is shifted along the x-axis, so does si.p
    si = pi.compute_surface_interaction(ray)
    ek.forward(ray.o.x)
    assert ek.allclose(ek.gradient(si.p), [1, 0, 0])

    # If the ray origin is shifted along the y-axis, so does si.p
    si = pi.compute_surface_interaction(ray)
    ek.forward(ray.o.y)
    assert ek.allclose(ek.gradient(si.p), [0, 1, 0])

    # If the ray origin is shifted along the z-axis, so does si.t
    si = pi.compute_surface_interaction(ray)
    ek.forward(ray.o.z)
    assert ek.allclose(ek.gradient(si.t), -1)

    # If the ray direction is shifted along the x-axis, so does si.p
    si = pi.compute_surface_interaction(ray)
    ek.forward(ray.d.x)
    assert ek.allclose(ek.gradient(si.p), [10, 0, 0])

    # If the ray origin is shifted toward the center of the disk, so does si.uv.x
    ray = Ray3f(Vector3f(0.9999999, 0.0, -10.0), Vector3f(0.0, 0.0, 1.0), 0, [])
    ek.set_requires_gradient(ray.o)
    si = shape.ray_intersect(ray)
    ek.forward(ray.o.x)
    assert ek.allclose(ek.gradient(si.uv), [1, 0])

    # If the ray origin is shifted tangent to the disk, si.uv.y moves by 1 / (2pi)
    si = shape.ray_intersect(ray)
    ek.forward(ray.o.y)
    assert ek.allclose(ek.gradient(si.uv), [0, 0.5 / ek.pi], atol=1e-5)

    # If the ray origin is shifted tangent to the disk, si.dp_dv will also have a component is x
    si = shape.ray_intersect(ray)
    ek.forward(ray.o.y)
    assert ek.allclose(ek.gradient(si.dp_dv), [-1, 0, 0])


def test05_differentiable_surface_interaction_ray_backward(variant_gpu_autodiff_rgb):
    from mitsuba.core import xml, Ray3f, Vector3f, UInt32

    shape = xml.load_dict({'type' : 'diskhole'})

    ray = Ray3f(Vector3f(-0.3, -0.3, -10.0), Vector3f(0.0, 0.0, 1.0), 0, [])
    pi = shape.ray_intersect_preliminary(ray)

    ek.set_requires_gradient(ray.o)

    # If si.p is shifted along the x-axis, so does the ray origin
    si = pi.compute_surface_interaction(ray)
    ek.backward(si.p.x)
    assert ek.allclose(ek.gradient(ray.o), [1, 0, 0])

    # If si.t is changed, so does the ray origin along the z-axis
    si = pi.compute_surface_interaction(ray)
    ek.backward(si.t)
    assert ek.allclose(ek.gradient(ray.o), [0, 0, -1])
