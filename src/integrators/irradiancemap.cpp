#include <enoki/stl.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <random>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class IrradianceMapIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    IrradianceMapIntegrator(const Properties &props) : Base(props) { }


    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
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
        Mask valid_ray = si.is_valid();
        EmitterPtr emitter = si.emitter(scene);

        for (int depth = 1;; ++depth) {

            // ---------------- Intersection with emitters ----------------

            if (any_or<true>(neq(emitter, nullptr)))
                result[active] += emission_weight * throughput * emitter->eval(si, active);

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
            Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);


            //if (likely(any_or<true>(active_e))) {
            //    auto [ds, emitter_val] = scene->sample_emitter_direction(
            //        si, sampler->next_2d(active_e), true, active_e);
            //    active_e &= neq(ds.pdf, 0.f);

            //    // Query the BSDF for that emitter-sampled direction
            //    Vector3f wo       = si.to_local(ds.d);
            //    Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
            //    bsdf_val          = si.to_world_mueller(bsdf_val, -wo, si.wi);

            //    // Determine density of sampling that same direction using BSDF
            //    // sampling
            //    Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);

            //    Mask active_irrad = true;
            //    active_irrad &= eq(bsdf_pdf, -1.0f);
            //    active_e ^= active_irrad;

            //    Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
            //    result[active_e] += mis * throughput * bsdf_val * emitter_val;

            //    result[active_irrad] += bsdf_val;
            //    
            //}

            if (likely(any_or<true>(active_e))) {
                //auto [ds, emitter_val] = scene->sample_emitter_direction(
                //    si, sampler->next_2d(active_e), true, active_e);
                //active_e &= neq(ds.pdf, 0.f);
                 
                // Query the BSDF for that emitter-sampled direction
                //Vector3f wo       = si.to_local(si.wi);
                Vector3f wo = 0;
                Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                //bsdf_val          = si.to_world_mueller(bsdf_val, -wo, si.wi);

                result[active_e] += bsdf_val;
            }
            break;
        }

        return { result, valid_ray };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("IrradianceMapIntegrator[\n"
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

MTS_IMPLEMENT_CLASS_VARIANT(IrradianceMapIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(IrradianceMapIntegrator, "Irradiance Map integrator");
NAMESPACE_END(mitsuba)