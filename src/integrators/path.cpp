#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path:

Path tracer (:monosp:`path`)
-------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). A value of 1 will only render directly visible light sources. 2 will lead
     to single-bounce (direct-only) illumination, and so on. (Default: -1)
 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start to use the
     *russian roulette* path termination criterion. (Default: 5)
 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This integrator implements a basic path tracer and is a **good default choice**
when there is no strong reason to prefer another method.

To use the path tracer appropriately, it is instructive to know roughly how
it works: its main operation is to trace many light paths using *random walks*
starting from the sensor. A single random walk is shown below, which entails
casting a ray associated with a pixel in the output image and searching for
the first visible intersection. A new direction is then chosen at the intersection,
and the ray-casting step repeats over and over again (until one of several
stopping criteria applies).

.. image:: ../images/integrator_path_figure.png
    :width: 95%
    :align: center

At every intersection, the path tracer tries to create a connection to
the light source in an attempt to find a *complete* path along which
light can flow from the emitter to the sensor. This of course only works
when there is no occluding object between the intersection and the emitter.

This directly translates into a category of scenes where
a path tracer can be expected to produce reasonable results: this is the case
when the emitters are easily "accessible" by the contents of the scene. For instance,
an interior scene that is lit by an area light will be considerably harder
to render when this area light is inside a glass enclosure (which
effectively counts as an occluder).

Like the :ref:`direct <integrator-direct>` plugin, the path tracer internally relies on multiple importance
sampling to combine BSDF and emitter samples. The main difference in comparison
to the former plugin is that it considers light paths of arbitrary length to compute
both direct and indirect illumination.

.. _sec-path-strictnormals:

.. Commented out for now
.. Strict normals
   --------------

.. Triangle meshes often rely on interpolated shading normals
   to suppress the inherently faceted appearance of the underlying geometry. These
   "fake" normals are not without problems, however. They can lead to paradoxical
   situations where a light ray impinges on an object from a direction that is
   classified as "outside" according to the shading normal, and "inside" according
   to the true geometric normal.

.. The :paramtype:`strict_normals` parameter specifies the intended behavior when such cases arise. The
   default (|false|, i.e. "carry on") gives precedence to information given by the shading normal and
   considers such light paths to be valid. This can theoretically cause light "leaks" through
   boundaries, but it is not much of a problem in practice.

.. When set to |true|, the path tracer detects inconsistencies and ignores these paths. When objects
   are poorly tesselated, this latter option may cause them to lose a significant amount of the
   incident radiation (or, in other words, they will look dark).

.. note:: This integrator does not handle participating media

 */

template <typename Float, typename Spectrum>
class PathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    PathIntegrator(const Properties &props) : Base(props) { }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                    const RayDifferential3f &ray_,
                                    const Medium * /* medium */,
                                    Float * /* aovs */,
                                    Mask active) const override {
       MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

       RayDifferential3f ray = ray_;

       // Tracks radiance scaling due to index of refraction changes
       Float eta(1.f);

       // MIS weight for intersected emitters (set by prev. iteration)
       Float emission_weight(1.f);

       Spectrum throughput(1.f), result(0.f);

       // ---------------------- First intersection ----------------------

       SurfaceInteraction3f si = scene->ray_intersect(ray, active);
       Mask valid_ray          = si.is_valid();
       EmitterPtr emitter      = si.emitter(scene);

       for (int depth = 1;; ++depth) {

           // ---------------- Intersection with emitters ----------------

           if (any_or<true>(neq(emitter, nullptr)))
               result[active] +=
                   emission_weight * throughput * emitter->eval(si, active);

           active &= si.is_valid();

           /* Russian roulette: try to keep path weights equal to one,
              while accounting for the solid angle compression at refractive
              index boundaries. Stop with at least some probability to avoid
              getting stuck (e.g. due to total internal reflection) */
           if (depth > m_rr_depth) {
               Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
               active &= sampler->next_1d(active) < q;
               throughput *= rcp(q);
           }

           // Stop if we've exceeded the number of requested bounces, or
           // if there are no more active lanes. Only do this latter check
           // in GPU mode when the number of requested bounces is infinite
           // since it causes a costly synchronization.
           if ((uint32_t) depth >= (uint32_t) m_max_depth ||
               ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active)))
               break;

           // --------------------- Emitter sampling ---------------------

           BSDFContext ctx;
           BSDFPtr bsdf = si.bsdf(ray);
           Mask active_e =
               active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

           if (likely(any_or<true>(active_e))) {
               auto [ds, emitter_val] = scene->sample_emitter_direction(
                   si, sampler->next_2d(active_e), true, active_e);
               active_e &= neq(ds.pdf, 0.f);

               // Query the BSDF for that emitter-sampled direction
               Vector3f wo       = si.to_local(ds.d);
               Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
               bsdf_val          = si.to_world_mueller(bsdf_val, -wo, si.wi);

               // Determine density of sampling that same direction using BSDF
               // sampling
               Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);

               Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
               result[active_e] += mis * throughput * bsdf_val * emitter_val;
           }

           // ----------------------- BSDF sampling ----------------------

           // Sample BSDF * cos(theta)
           auto [bs, bsdf_val] =
               bsdf->sample(ctx, si, sampler->next_1d(active),
                            sampler->next_2d(active), active);
           bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

           throughput = throughput * bsdf_val;
           active &= any(neq(depolarize(throughput), 0.f));
           if (none_or<false>(active))
               break;

           eta *= bs.eta;

           // Intersect the BSDF ray against the scene geometry
           ray                          = si.spawn_ray(si.to_world(bs.wo));
           SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

           /* Determine probability of having sampled that same
              direction using emitter sampling. */
           emitter = si_bsdf.emitter(scene, active);
           DirectionSample3f ds(si_bsdf, si);
           ds.object = emitter;

           if (any_or<true>(neq(emitter, nullptr))) {
               Float emitter_pdf =
                   select(neq(emitter, nullptr) &&
                              !has_flag(bs.sampled_type, BSDFFlags::Delta),
                          scene->pdf_emitter_direction(si, ds), 0.f);

               emission_weight = mis_weight(bs.pdf, emitter_pdf);
           }

           si = std::move(si_bsdf);
       }

       return { result, valid_ray };
    }

    std::pair<Spectrum, Mask> sample_window(
        const Scene *scene, Sampler *sampler, const RayDifferential3f &ray_main_,
        const RayDifferential3f& ray_sub_, const Medium * /* medium */,
        Float * /* aovs */, Mask active_main = true, Mask active_sub = true) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active_sub);

        RayDifferential3f ray_main = ray_main_;
        RayDifferential3f ray_sub  = ray_sub_;
        // ---------------------- First intersection ----------------------

        SurfaceInteraction3f si_main = scene->ray_intersect(ray_main, active_main);
        Mask valid_ray_main          = si_main.is_valid();
        active_main &= si_main.is_valid();
        BSDFPtr bsdf_main = si_main.bsdf(ray_main);
        Mask active_e_main = active_main && has_flag(bsdf_main->flags(), BSDFFlags::Smooth);

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        // MIS weight for intersected emitters (set by prev. iteration)
        Float emission_weight(1.f);

        Spectrum throughput(1.f), result(0.f);

        // ---------------------- First intersection ----------------------
        SurfaceInteraction3f si = scene->ray_intersect(ray_sub, active_sub);
        Mask valid_ray_sub          = si.is_valid();
        EmitterPtr emitter      = si.emitter(scene);

        active_sub &= active_main;
        valid_ray_sub &= valid_ray_main;

        for (int depth = 1;; ++depth) {

            // ---------------- Intersection with emitters ----------------

            if (any_or<true>(neq(emitter, nullptr)))
                result[active_sub] +=
                    emission_weight * throughput * emitter->eval(si, active_sub);

            active_sub &= si.is_valid();

            /* Russian roulette: try to keep path weights equal to one,
               while accounting for the solid angle compression at refractive
               index boundaries. Stop with at least some probability to avoid
               getting stuck (e.g. due to total internal reflection) */
            if (depth > m_rr_depth) {
                Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                active_sub &= sampler->next_1d(active_sub) < q;
                throughput *= rcp(q);
            }

            // Stop if we've exceeded the number of requested bounces, or
            // if there are no more active lanes. Only do this latter check
            // in GPU mode when the number of requested bounces is infinite
            // since it causes a costly synchronization.
            if ((uint32_t) depth >= (uint32_t) m_max_depth ||
                ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active_sub)))
                break;

            // --------------------- Emitter sampling ---------------------

            BSDFContext ctx;
            BSDFPtr bsdf = si.bsdf(ray_sub);
            Mask active_e =
                active_sub && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            active_e &= active_e_main;
            if (likely(any_or<true>(active_e))) {
                auto [ds, emitter_val] = scene->sample_emitter_direction(
                    si, sampler->next_2d(active_e), true, active_e);
                active_e &= neq(ds.pdf, 0.f);

                // Query the BSDF for that emitter-sampled direction
                Vector3f wo       = si.to_local(ds.d);
                Spectrum bsdf_val = bsdf->eval_window(ctx, si_main, si, wo, active_e_main, active_e);
                bsdf_val          = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Determine density of sampling that same direction using BSDF
                // sampling
                Float bsdf_pdf = bsdf->pdf_window(ctx, si_main, si, wo, active_e_main, active_e);

                Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                result[active_e] += mis * throughput * bsdf_val * emitter_val;
            }

            // ----------------------- BSDF sampling ----------------------

            // Sample BSDF * cos(theta)
            auto [bs, bsdf_val] =
                bsdf->sample_window(ctx, si_main, si, sampler->next_1d(active_sub),
                             sampler->next_2d(active_sub), active_main, active_sub);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            throughput = throughput * bsdf_val;
            active_sub &= any(neq(depolarize(throughput), 0.f));
            if (none_or<false>(active_sub))
                break;

            eta *= bs.eta;

            // Intersect the BSDF ray against the scene geometry
            ray_sub                      = si.spawn_ray(si.to_world(bs.wo));
            SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray_sub, active_sub);

            /* Determine probability of having sampled that same
               direction using emitter sampling. */
            emitter = si_bsdf.emitter(scene, active_sub);
            DirectionSample3f ds(si_bsdf, si);
            ds.object = emitter;

            if (any_or<true>(neq(emitter, nullptr))) {
                Float emitter_pdf =
                    select(neq(emitter, nullptr) &&
                               !has_flag(bs.sampled_type, BSDFFlags::Delta),
                           scene->pdf_emitter_direction(si, ds), 0.f);

                emission_weight = mis_weight(bs.pdf, emitter_pdf);
            }

            si = std::move(si_bsdf);
        }

        return { result, valid_ray_sub };
        //int subray_cnt = ray_w.size();
        //result_list.resize(subray_cnt);
        //for (int i = 0; i < subray_cnt; ++i) {
        //    // Tracks radiance scaling due to index of refraction changes
        //    Float eta(1.f);

        //    // MIS weight for intersected emitters (set by prev. iteration)
        //    Float emission_weight(1.f);

        //    Spectrum throughput(1.f);

        //    Mask active_t = true;
        //    RayDifferential3f ray = ray_w[i];

        //    SurfaceInteraction3f si = scene->ray_intersect(ray, active_t);
        //    EmitterPtr emitter = si.emitter(scene);

        //    valid_ray &= si.is_valid();
        //    active_t &= active;

        //    result_list[i] = Spectrum(0.f);
        //    for (int depth = 1;; ++depth) {

        //        // ---------------- Intersection with emitters ----------------

        //        if (any_or<true>(neq(emitter, nullptr)))
        //            result_list[i][active_t] +=
        //                emission_weight * throughput * emitter->eval(si, active_t);

        //        active_t &= si.is_valid();

        //        /* Russian roulette: try to keep path weights equal to one,
        //           while accounting for the solid angle compression at refractive
        //           index boundaries. Stop with at least some probability to avoid
        //           getting stuck (e.g. due to total internal reflection) */
        //        if (depth > m_rr_depth) {
        //            Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
        //            active_t &= sampler->next_1d(active_t) < q;
        //            throughput *= rcp(q);
        //        }

        //        // Stop if we've exceeded the number of requested bounces, or
        //        // if there are no more active lanes. Only do this latter check
        //        // in GPU mode when the number of requested bounces is infinite
        //        // since it causes a costly synchronization.
        //        if ((uint32_t) depth >= (uint32_t) m_max_depth ||
        //            ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active_t)))
        //            break;

        //        // --------------------- Emitter sampling ---------------------

        //        BSDFContext ctx;
        //        BSDFPtr bsdf = si.bsdf(ray);
        //        Mask active_e =
        //            active_t && has_flag(bsdf->flags(), BSDFFlags::Smooth);

        //        if (likely(any_or<true>(active_e))) {
        //            auto [ds, emitter_val] = scene->sample_emitter_direction(
        //                si, sampler->next_2d(active_e), true, active_e);
        //            active_e &= neq(ds.pdf, 0.f);

        //            // Query the BSDF for that emitter-sampled direction
        //            Vector3f wo       = si.to_local(ds.d);
        //            Spectrum bsdf_val = bsdf->eval_window(ctx, si_m, si, wo, active_e_m, active_e);
        //            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);
        //            result_list[i][active_e] += throughput * bsdf_val * emitter_val;
        //            break;

        //            // Determine density of sampling that same direction using BSDF
        //            // sampling
        //            Float bsdf_pdf = bsdf->pdf_window(ctx, si_m, si, wo, active_e_m, active_e);

        //            Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
        //            result_list[i][active_e] += mis * throughput * bsdf_val * emitter_val;
        //        }

        //        // ----------------------- BSDF sampling ----------------------

        //        // Sample BSDF * cos(theta)
        //        auto [bs, bsdf_val] =
        //            bsdf->sample_window(ctx, si_m, si, sampler->next_1d(active_t),
        //                         sampler->next_2d(active_t), active_e_m, active_t);
        //        bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

        //        throughput = throughput * bsdf_val;
        //        active_t &= any(neq(depolarize(throughput), 0.f));
        //        if (none_or<false>(active_t))
        //            break;

        //        eta *= bs.eta;

        //        // Intersect the BSDF ray against the scene geometry
        //        ray                          = si.spawn_ray(si.to_world(bs.wo));
        //        SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active_t);

        //        /* Determine probability of having sampled that same
        //           direction using emitter sampling. */
        //        emitter = si_bsdf.emitter(scene, active_t);
        //        DirectionSample3f ds(si_bsdf, si);
        //        ds.object = emitter;

        //        if (any_or<true>(neq(emitter, nullptr))) {
        //            Float emitter_pdf =
        //                select(neq(emitter, nullptr) &&
        //                           !has_flag(bs.sampled_type, BSDFFlags::Delta),
        //                       scene->pdf_emitter_direction(si, ds), 0.f);

        //            emission_weight = mis_weight(bs.pdf, emitter_pdf);
        //        }

        //        si = std::move(si_bsdf);
        //    }

        //}

        //return { result_list, valid_ray };
    }
    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i\n"
            "]", m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(PathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PathIntegrator, "Path Tracer integrator");
NAMESPACE_END(mitsuba)
