#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/rfilter.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>

#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-diffuse:

Smooth diffuse material (:monosp:`diffuse`)
-------------------------------------------

.. pluginparameters::

 * - reflectance
   - |spectrum| or |texture|
   - Specifies the diffuse albedo of the material (Default: 0.5)

The smooth diffuse material (also referred to as *Lambertian*)
represents an ideally diffuse material with a user-specified amount of
reflectance. Any received illumination is scattered so that the surface
looks the same independently of the direction of observation.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/bsdf_diffuse_plain.jpg
   :caption: Homogeneous reflectance
.. subfigure:: ../../resources/data/docs/images/render/bsdf_diffuse_textured.jpg
   :caption: Textured reflectance
.. subfigend::
   :label: fig-diffuse

Apart from a homogeneous reflectance value, the plugin can also accept
a nested or referenced texture map to be used as the source of reflectance
information, which is then mapped onto the shape based on its UV
parameterization. When no parameters are specified, the model uses the default
of 50% reflectance.

Note that this material is one-sided---that is, observed from the
back side, it will be completely black. If this is undesirable,
consider using the :ref:`twosided <bsdf-twosided>` BRDF adapter plugin.
The following XML snippet describes a diffuse material,
whose reflectance is specified as an sRGB color:

.. code-block:: xml
    :name: diffuse-srgb

    <bsdf type="diffuse">
        <rgb name="reflectance" value="0.2, 0.25, 0.7"/>
    </bsdf>

Alternatively, the reflectance can be textured:

.. code-block:: xml
    :name: diffuse-texture

    <bsdf type="diffuse">
        <texture type="bitmap" name="reflectance">
            <string name="filename" value="wood.jpg"/>
        </texture>
    </bsdf>

*/

template <typename Float, typename Spectrum>
class SmoothDiffuse final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture, ReconstructionFilter)
        
    struct CatchBitmap {
        DynamicBuffer<Float> m_data;
        DynamicBuffer<Float> m_weights;
        ScalarVector2i m_resolution;
        enoki::divisor<int32_t> m_inv_resolution_x;
        enoki::divisor<int32_t> m_inv_resolution_y;

        ref<Bitmap> m_bitmap = nullptr;

        ref<ReconstructionFilter> m_filter = nullptr;
        Float *m_weights_x, *m_weights_y;
        int m_border_size;
        bool border = false;
        CatchBitmap(int w, int h) : m_resolution(ScalarVector2i(w, h)),
              m_inv_resolution_x(w),
              m_inv_resolution_y(h)
        { 
            //m_dtmp  = DynamicBuffer<Float>::zero_(hprod(m_resolution) * 3);
            m_filter = PluginManager::instance()->create_object<ReconstructionFilter>(Properties("gaussian"));
            int filter_size = (int) std::ceil(2 * m_filter->radius()) + 1;
            m_weights_x = new Float[2 * filter_size];
            m_weights_y = m_weights_x + filter_size;

            m_border_size = border ? m_filter->border_size() : 0;
            size_t size   = 3 * hprod(m_resolution + 2 * m_border_size);
            m_data = DynamicBuffer<Float>::zero_(size);
            m_weights = DynamicBuffer<Float>::zero_(hprod(m_resolution + 2 * m_border_size));

            const Bitmap::Vector2u bsize((uint32_t) w, (uint32_t) h);
            m_bitmap = new Bitmap(Bitmap::PixelFormat::RGB, Struct::Type::Float32, bsize, 3u, nullptr);
        }
        ~CatchBitmap() { 
            if (m_weights_x) {
                delete[] m_weights_x;
            }
        }
        template <typename T>
        T wrap(const T &value) const{
            T div = T(m_inv_resolution_x(value.x()),
                      m_inv_resolution_y(value.y())),
              mod = value - div * m_resolution;

            masked(mod, mod < 0) += T(m_resolution);

            return mod;
        }
        void catch_spec(const Spectrum &spec, const SurfaceInteraction3f &si, Mask active) {
            Point2f uv = si.uv;
            uv *= m_resolution;
//#define GEN_BASE
#ifdef GEN_BASE

            Vector2i uv_i = floor2int<Vector2i>(uv);
            Vector2i uv_i_w = wrap(uv_i);

            Int32 index = uv_i.x() + uv_i.y() * m_resolution.x();
            //m_dtmp *= 0;
            //scatter(m_dtmp, spec, index, active);
            //m_data += m_dtmp;
             scatter(m_data, spec, index, active);
#else

            //uv = fmadd(uv, m_resolution, -.5f);
            //Vector2i uv_i = floor2int<Vector2i>(uv);
            //Int21 uv_i_w  = Int21(uv_i.x(), uv_i.y());

            //Int1 index = uv_i_w.x() + uv_i_w.y() * m_resolution.x();
            //scatter(m_data, spec, index.x(), active);
             
            std::vector<Float> aovs(3);
            aovs[0] = spec.x();
            aovs[1] = spec.y();
            aovs[2] = spec.z();

            ScalarFloat filter_radius = m_filter->radius();
            ScalarVector2i size = m_resolution + 2 * m_border_size;
            Point2f pos = uv - (0 - m_border_size + .5f);

            Point2u lo = Point2u(max(ceil2int<Point2i>(pos - filter_radius), 0))
                    , hi = Point2u(min(floor2int<Point2i>(pos + filter_radius), size - 1));
            uint32_t n = ceil2int<uint32_t>((filter_radius - 2.f * math::RayEpsilon<ScalarFloat>) * 2.f);

            Point2f base = lo - pos;
            for (uint32_t i = 0; i < n; ++i) {
                Point2f p = base + i;
                if constexpr (!is_cuda_array_v<Float>) {
                    m_weights_x[i] = m_filter->eval_discretized(p.x(), active);
                    m_weights_y[i] = m_filter->eval_discretized(p.y(), active);
                } else {
                    m_weights_x[i] = m_filter->eval(p.x(), active);
                    m_weights_y[i] = m_filter->eval(p.y(), active);
                }
            }

            Float wx(0), wy(0);
            for (uint32_t i = 0; i < n; ++i) {
                wx += m_weights_x[i];
                wy += m_weights_y[i];
            }
            Float factor = rcp(wx * wy);
            for (uint32_t i = 0; i < n; ++i)
                m_weights_x[i] *= factor;

            Float *value = aovs.data();
            ENOKI_NOUNROLL for (uint32_t yr = 0; yr < n; ++yr) {
                UInt32 y     = lo.y() + yr;
                Mask enabled = active && y <= hi.y();

                ENOKI_NOUNROLL for (uint32_t xr = 0; xr < n; ++xr) {
                    UInt32 x = lo.x() + xr,
                        offset = 3 * (y * size.x() + x);
                    Float weight  = m_weights_y[yr] * m_weights_x[xr];

                    enabled &= x <= hi.x();

                    scatter_add(m_weights, weight, (y * size.x() + x), enabled);
                    ENOKI_NOUNROLL for (uint32_t k = 0; k < 3; ++k) {
                        scatter_add(m_data, value[k] * weight, offset + k, enabled);
                    }
                }
            }           
#endif
        }
    };

    SmoothDiffuse(const Properties &props) : Base(props) {
        m_reflectance = props.texture<Texture>("reflectance", .5f);
        m_catch_irradiance = props.bool_("catch_irradiance", false);
        m_irradiance_width = props.int_("irradiance_width", 0);
        m_irradiance_height = props.int_("irradiance_height", 0);
        m_irradiance_filename = props.string("irradiance_filename", "");
        if (m_catch_irradiance) {
            m_catch_bitmap = new CatchBitmap(m_irradiance_width, m_irradiance_height);
        }

        if (props.has_property("irradiance")) {
            m_irradiance   = props.texture<Texture>("irradiance");
        }
        m_flags = BSDFFlags::DiffuseReflection | BSDFFlags::FrontSide;
        m_components.push_back(m_flags);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /* sample1 */,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs = zero<BSDFSample3f>();

        active &= cos_theta_i > 0.f;
        if (unlikely(none_or<false>(active) ||
                     !ctx.is_enabled(BSDFFlags::DiffuseReflection)))
            return { bs, 0.f };

        bs.wo = warp::square_to_cosine_hemisphere(sample2);
        bs.pdf = warp::square_to_cosine_hemisphere_pdf(bs.wo);
        bs.eta = 1.f;
        bs.sampled_type = +BSDFFlags::DiffuseReflection;
        bs.sampled_component = 0;

        UnpolarizedSpectrum value = m_reflectance->eval(si, active);

        return { bs, select(active && bs.pdf > 0.f, unpolarized<Spectrum>(value), 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::DiffuseReflection))
            return 0.f;
        
        if (m_irradiance != nullptr) {
            UnpolarizedSpectrum value = m_irradiance->eval(si, active);
            //return select(active, unpolarized<Spectrum>(value), 0.f);
            return value;
        }

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        UnpolarizedSpectrum value =
            m_reflectance->eval(si, active) * math::InvPi<Float> * cos_theta_o;

        return select(active, unpolarized<Spectrum>(value), 0.f);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::DiffuseReflection))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo);

        return select(cos_theta_i > 0.f && cos_theta_o > 0.f, pdf, 0.f);
    }

    void catch_irradiance(const BSDFContext &ctx,
                                                 const SurfaceInteraction3f &si,
                                                 const Vector3f &wo,
                                                 const Spectrum &emitter_val,
                                                 Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::DiffuseReflection))
            return;

        if (!m_catch_irradiance) {
            return;
        }

        //Float cos_theta_i = Frame3f::cos_theta(si.wi),
        //      cos_theta_o = Frame3f::cos_theta(wo);

        //active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        Spectrum irradiance_val =
            select(active, emitter_val, 0.f);

        // TODO write in a buffer
        m_catch_bitmap->catch_spec(irradiance_val, si, active);

        //std::cout << "aha! diffuse catch_irradiance\n";
    }

    void acc_irradiance(const float factor) const override {
        if (!m_catch_irradiance || factor <= 0) {
            return;
        }
  
        auto m_data  = m_catch_bitmap->m_data.managed();
        auto m_weights  = m_catch_bitmap->m_weights.managed();
        const ScalarFloat *fptr = m_data.data();
        const ScalarFloat *wptr = m_weights.data();
        float *cdata            = new float[hprod(m_catch_bitmap->m_resolution) * 3];
        for (int i = 0, pixel_count = hprod(m_catch_bitmap->m_resolution);
             i < pixel_count; ++i) {
            ScalarColor3f fvalue = load_unaligned<ScalarColor3f>(fptr);
            ScalarFloat wvalue   = load_unaligned<ScalarFloat>(wptr);
            if (wvalue > 0) {
                cdata[i * 3 + 0] = fvalue.x() * factor / wvalue;
                cdata[i * 3 + 1] = fvalue.y() * factor / wvalue;
                cdata[i * 3 + 2] = fvalue.z() * factor / wvalue;
            }
            fptr += 3;
            ++wptr;
        }
        uint8_t *udata = (uint8_t *) cdata;
        const Bitmap::Vector2u size((uint32_t) m_irradiance_width,
                                    (uint32_t) m_irradiance_height);

        ref<Bitmap> irrad_bitmap = new Bitmap(Bitmap::PixelFormat::RGB, Struct::Type::Float32, size, 3u, udata);
        m_catch_bitmap->m_bitmap->accumulate(irrad_bitmap, 0, 0, size);
        // clear
        m_catch_bitmap->m_data *= 0;
        m_catch_bitmap->m_weights *= 0;

        delete cdata;
        //std::cout << "aha! save diffuse irradiance\n";
    }

    void save_irradiance() const override {
        if (!m_catch_irradiance || m_catch_bitmap == nullptr || m_catch_bitmap->m_bitmap == nullptr) {
            return;
        }

        m_catch_bitmap->m_bitmap->write(fs::path(m_irradiance_filename));
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("reflectance", m_reflectance.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SmoothDiffuse[" << std::endl
            << "  reflectance = " << string::indent(m_reflectance) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_reflectance;
    ref<Texture> m_irradiance;

    //ref<Bitmap> m_catch_irradiance;

    bool m_catch_irradiance = false;
    int m_irradiance_width, m_irradiance_height;
    std::string m_irradiance_filename;

    CatchBitmap* m_catch_bitmap;
    
};

MTS_IMPLEMENT_CLASS_VARIANT(SmoothDiffuse, BSDF)
MTS_EXPORT_PLUGIN(SmoothDiffuse, "Smooth diffuse material")
NAMESPACE_END(mitsuba)
