#include <mitsuba/core/string.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/texture.h>

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/fresolver.h>
#include <array>
#include <mitsuba/core/rfilter.h>
#include <mitsuba/core/plugin.h>

NAMESPACE_BEGIN(mitsuba)

/**!
.. _bsdf-roughconductor:

Rough conductor material (:monosp:`roughconductor`)
---------------------------------------------------

.. pluginparameters::

 * - material
   - |string|
   - Name of the material preset, see :num:`conductor-ior-list`. (Default: none)
 * - eta, k
   - |spectrum| or |texture|
   - Real and imaginary components of the material's index of refraction. (Default: based on the value of :monosp:`material`)
 * - specular_reflectance
   - |spectrum| or |texture|
   - Optional factor that can be used to modulate the specular reflection component.
     Note that for physical realism, this parameter should never be touched. (Default: 1.0)

 * - distribution
   - |string|
   - Specifies the type of microfacet normal distribution used to model the surface roughness.

     - :monosp:`beckmann`: Physically-based distribution derived from Gaussian random surfaces.
       This is the default.
     - :monosp:`ggx`: The GGX :cite:`Walter07Microfacet` distribution (also known as Trowbridge-Reitz
       :cite:`Trowbridge19975Average` distribution) was designed to better approximate the long
       tails observed in measurements of ground surfaces, which are not modeled by the Beckmann
       distribution.
 * - alpha, alpha_u, alpha_v
   - |texture| or |float|
   - Specifies the roughness of the unresolved surface micro-geometry along the tangent and
     bitangent directions. When the Beckmann distribution is used, this parameter is equal to the
     **root mean square** (RMS) slope of the microfacets. :monosp:`alpha` is a convenience
     parameter to initialize both :monosp:`alpha_u` and :monosp:`alpha_v` to the same value. (Default: 0.1)
 * - sample_visible
   - |bool|
   - Enables a sampling technique proposed by Heitz and D'Eon :cite:`Heitz1014Importance`, which
     focuses computation on the visible parts of the microfacet normal distribution, considerably
     reducing variance in some cases. (Default: |true|, i.e. use visible normal sampling)

This plugin implements a realistic microfacet scattering model for rendering
rough conducting materials, such as metals.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/bsdf_roughconductor_copper.jpg
   :caption: Rough copper (Beckmann, :math:`\alpha=0.1`)
.. subfigure:: ../../resources/data/docs/images/render/bsdf_roughconductor_anisotropic_aluminium.jpg
   :caption: Vertically brushed aluminium (Anisotropic Beckmann, :math:`\alpha_u=0.05,\ \alpha_v=0.3`)
.. subfigure:: ../../resources/data/docs/images/render/bsdf_roughconductor_textured_carbon.jpg
   :caption: Carbon fiber using two inverted checkerboard textures for ``alpha_u`` and ``alpha_v``
.. subfigend::
    :label: fig-bsdf-roughconductor


Microfacet theory describes rough surfaces as an arrangement of unresolved
and ideally specular facets, whose normal directions are given by a
specially chosen *microfacet distribution*. By accounting for shadowing
and masking effects between these facets, it is possible to reproduce the
important off-specular reflections peaks observed in real-world measurements
of such materials.

This plugin is essentially the *roughened* equivalent of the (smooth) plugin
:ref:`conductor <bsdf-conductor>`. For very low values of :math:`\alpha`, the two will
be identical, though scenes using this plugin will take longer to render
due to the additional computational burden of tracking surface roughness.

The implementation is based on the paper *Microfacet Models
for Refraction through Rough Surfaces* by Walter et al.
:cite:`Walter07Microfacet` and it supports two different types of microfacet
distributions.

To facilitate the tedious task of specifying spectrally-varying index of
refraction information, this plugin can access a set of measured materials
for which visible-spectrum information was publicly available
(see the corresponding table in the :ref:`conductor <bsdf-conductor>` reference).

When no parameters are given, the plugin activates the default settings,
which describe a 100% reflective mirror with a medium amount of roughness modeled
using a Beckmann distribution.

To get an intuition about the effect of the surface roughness parameter
:math:`\alpha`, consider the following approximate classification: a value of
:math:`\alpha=0.001-0.01` corresponds to a material with slight imperfections
on an otherwise smooth surface finish, :math:`\alpha=0.1` is relatively rough,
and :math:`\alpha=0.3-0.7` is **extremely** rough (e.g. an etched or ground
finish). Values significantly above that are probably not too realistic.


The following XML snippet describes a material definition for brushed aluminium:

.. code-block:: xml
    :name: lst-roughconductor-aluminium

    <bsdf type="roughconductor">
        <string name="material" value="Al"/>
        <string name="distribution" value="ggx"/>
        <float name="alphaU" value="0.05"/>
        <float name="alphaV" value="0.3"/>
    </bsdf>

Technical details
*****************

All microfacet distributions allow the specification of two distinct
roughness values along the tangent and bitangent directions. This can be
used to provide a material with a *brushed* appearance. The alignment
of the anisotropy will follow the UV parameterization of the underlying
mesh. This means that such an anisotropic material cannot be applied to
triangle meshes that are missing texture coordinates.

Since Mitsuba 0.5.1, this plugin uses a new importance sampling technique
contributed by Eric Heitz and Eugene D'Eon, which restricts the sampling
domain to the set of visible (unmasked) microfacet normals. The previous
approach of sampling all normals is still available and can be enabled
by setting :monosp:`sample_visible` to :monosp:`false`. However this will lead
to significantly slower convergence.

When using this plugin, you should ideally compile Mitsuba with support for
spectral rendering to get the most accurate results. While it also works
in RGB mode, the computations will be more approximate in nature.
Also note that this material is one-sided---that is, observed from the
back side, it will be completely black. If this is undesirable,
consider using the :ref:`twosided <bsdf-twosided>` BRDF adapter.

In *polarized* rendering modes, the material automatically switches to a polarized
implementation of the underlying Fresnel equations.

 */

template <typename Float, typename Spectrum>
class RoughConductor final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture, MicrofacetDistribution, ReconstructionFilter)

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

        CatchBitmap(int w, int h)
            : m_resolution(ScalarVector2i(w, h)), m_inv_resolution_x(w),
              m_inv_resolution_y(h) {
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
        template <typename T> T wrap(const T &value) const {
            T div = T(m_inv_resolution_x(value.x()),
                      m_inv_resolution_y(value.y())),
              mod = value - div * m_resolution;

            masked(mod, mod < 0) += T(m_resolution);

            return mod;
        }
        void catch_spec(const Spectrum& spec, const SurfaceInteraction3f& si,
            Mask active) {
            Point2f uv = si.uv;
            uv *= m_resolution;
//#define GEN_BASE
#ifdef GEN_BASE

            Vector2i uv_i   = floor2int<Vector2i>(uv);
            Vector2i uv_i_w = wrap(uv_i);

            Int32 index = uv_i.x() + uv_i.y() * m_resolution.x();
            // m_dtmp *= 0;
            // scatter(m_dtmp, spec, index, active);
            // m_data += m_dtmp;
            scatter(m_data, spec, index, active);
#else
            std::vector<Float> aovs(3);
            aovs[0] = spec.x();
            aovs[1] = spec.y();
            aovs[2] = spec.z();

            ScalarFloat filter_radius = m_filter->radius();
            ScalarVector2i size = m_resolution + 2 * m_border_size;
            Point2f pos = uv - (0 - m_border_size + .5f);
            Point2u lo =
                        Point2u(max(ceil2int<Point2i>(pos - filter_radius), 0)),
                    hi = Point2u(
                        min(floor2int<Point2i>(pos + filter_radius), size - 1));
            uint32_t n = ceil2int<uint32_t>(
                (filter_radius - 2.f * math::RayEpsilon<ScalarFloat>) *2.f);

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
                    UInt32 x = lo.x() + xr, offset = 3 * (y * size.x() + x);
                    Float weight = m_weights_y[yr] * m_weights_x[xr];

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

    RoughConductor(const Properties &props) : Base(props) {
        std::string material = props.string("material", "none");
        if (props.has_property("eta") || material == "none") {
            m_eta = props.texture<Texture>("eta", 0.f);
            m_k   = props.texture<Texture>("k",   1.f);
            if (material != "none")
                Throw("Should specify either (eta, k) or material, not both.");
        } else {
            std::tie(m_eta, m_k) = complex_ior_from_file<Spectrum, Texture>(props.string("material", "Cu"));
        }

        if (props.has_property("distribution")) {
            std::string distr = string::to_lower(props.string("distribution"));
            if (distr == "beckmann")
                m_type = MicrofacetType::Beckmann;
            else if (distr == "ggx")
                m_type = MicrofacetType::GGX;
            else
                Throw("Specified an invalid distribution \"%s\", must be "
                      "\"beckmann\" or \"ggx\"!", distr.c_str());
        } else {
            m_type = MicrofacetType::Beckmann;
        }

        m_sample_visible = props.bool_("sample_visible", true);

        if (props.has_property("alpha_u") || props.has_property("alpha_v")) {
            if (!props.has_property("alpha_u") || !props.has_property("alpha_v"))
                Throw("Microfacet model: both 'alpha_u' and 'alpha_v' must be specified.");
            if (props.has_property("alpha"))
                Throw("Microfacet model: please specify"
                      "either 'alpha' or 'alpha_u'/'alpha_v'.");
            m_alpha_u = props.texture<Texture>("alpha_u");
            m_alpha_v = props.texture<Texture>("alpha_v");
        } else {
            m_alpha_u = m_alpha_v = props.texture<Texture>("alpha", 0.1f);
        }

        if (props.has_property("specular_reflectance"))
            m_specular_reflectance = props.texture<Texture>("specular_reflectance", 1.f);

        m_fresnel_shlick = props.bool_("fresnel_shlick", false);
        if (props.has_property("f0")) {
            m_f0 = props.texture<Texture>("f0", 0.f);
        }

        if (props.has_property("f90")) {
            m_f90 = props.texture<Texture>("f90", 1.0f);
        }

        m_ibl = props.bool_("ibl", false);
        if (m_ibl) {
            FileResolver *fs = Thread::thread()->file_resolver();
            std::string name  = props.string("brdflut");            
            fs::path file_path = fs->resolve(name);
            ref<Bitmap> bitmap = new Bitmap(file_path);

            //bitmap = bitmap->convert(Bitmap::PixelFormat::RGBA, struct_type_v<ScalarFloat>, false);
            bitmap = bitmap->convert(Bitmap::PixelFormat::RGB, struct_type_v<ScalarFloat>, false);
            m_brdflut.resolution = ScalarVector2i(bitmap->size());
            m_brdflut.data = DynamicBuffer<Float>::copy(bitmap->data(), hprod(m_brdflut.resolution) * 3);


            std::string pre_envmap_root = props.string("prefiltered_envmap_root");

            m_ibl_multiview = props.bool_("ibl_multiview", false);
            if (m_ibl_multiview) {
                for (int j = 0; j < 4; ++j) {
                    for (int i = 0; i < 41; ++i) {
                        int tagi = i / 40.0 * 1000;
                        std::string tag = std::to_string(tagi);
                        if (tag.size() < 2) tag = "000" + tag;
                        else if (tag.size() < 3) tag = "00" + tag;
                        else if (tag.size() < 4) tag = "0" + tag;
                        std::string name = (i == 40 ? (std::to_string(j) + "_env1000.exr") : (std::to_string(j) +"_env" + tag + ".exr"));
                        fs::path file_path = fs->resolve(pre_envmap_root + name);
                        ref<Bitmap> bitmap = new Bitmap(file_path);
                        bitmap = bitmap->convert(Bitmap::PixelFormat::RGB, struct_type_v<ScalarFloat>, false);
                        m_prefiltered_envmap.multiview_reso_list[j * 41 + i] = bitmap->size();
                        m_prefiltered_envmap.multiview_data_list[j * 41 + i] = DynamicBuffer<Float>::copy(bitmap->data(), hprod(m_prefiltered_envmap.multiview_reso_list[j * 41 + i]) * 3);
                    }
                }
            } else {
                //for (int i = 0; i < 11; ++i) {
                //    int tagi = i / 10.0 * 10;
                //    std::string tag = std::to_string(tagi);
                //    std::string name = (i == 10 ? "env10.exr" : ("env0" + tag + ".exr")); 
                //    fs::path file_path = fs->resolve(pre_envmap_root + name); 
                //    ref<Bitmap> bitmap = new Bitmap(file_path);

                for (int i = 0; i < 41; ++i) {
                    int tagi = i / 40.0 * 1000;
                    std::string tag = std::to_string(tagi);
                    if (tag.size() < 2) tag = "00" + tag;
                    else if (tag.size() < 3) tag = "0" + tag;
                    std::string name = (i == 40 ? "env1000.exr" : ("env0" + tag + ".exr"));
                    fs::path file_path = fs->resolve(pre_envmap_root + name);
                    ref<Bitmap> bitmap = new Bitmap(file_path);
#ifdef PREFILTERED
                    bitmap = bitmap->convert(Bitmap::PixelFormat::RGBA, struct_type_v<ScalarFloat>, false);
                    m_prefiltered_envmap.resolution_list[i] = bitmap->size();
                    m_prefiltered_envmap.data_list[i] = DynamicBuffer<Float>::copy(bitmap->data(), hprod(m_prefiltered_envmap.resolution_list[i]) * 4);
#else
                    bitmap = bitmap->convert(Bitmap::PixelFormat::RGB, struct_type_v<ScalarFloat>, false);
                    m_prefiltered_envmap.resolution_list[i] = bitmap->size();
                    m_prefiltered_envmap.data_list[i] = DynamicBuffer<Float>::copy(bitmap->data(), hprod(m_prefiltered_envmap.resolution_list[i]) * 3);
#endif
                }
            }
            m_prefiltered_envmap.scale = props.float_("envmap_scale", 1.f);
            if (props.has_property("envmap_to_world")) {
                m_prefiltered_envmap.world_transform = props.animated_transform("envmap_to_world", ScalarTransform4f()).get();
            }
        }

        m_catch_irradiance  = props.bool_("catch_irradiance", false);
        m_irradiance_width  = props.int_("irradiance_width", 0);
        m_irradiance_height   = props.int_("irradiance_height", 0);
        m_irradiance_filename      = props.string("irradiance_filename", "");
        m_envmap_scale = props.float_("envmap_scale", 1.f);
        if (m_catch_irradiance) {
            m_catch_bitmap = new CatchBitmap(m_irradiance_width, m_irradiance_height);
        }

        m_forward = props.bool_("forward", false);
        

        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide;
        if (m_alpha_u != m_alpha_v)
            m_flags = m_flags | BSDFFlags::Anisotropic;

        m_components.clear();
        m_components.push_back(m_flags);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /* sample1 */,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        BSDFSample3f bs = zero<BSDFSample3f>();
        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || none_or<false>(active)))
            return { bs, 0.f };

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type,
                                     m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        // Sample M, the microfacet normal
        Normal3f m;
        std::tie(m, bs.pdf) = distr.sample(si.wi, sample2);

        // Perfect specular reflection based on the microfacet normal
        bs.wo = reflect(si.wi, m);
        bs.eta = 1.f;
        bs.sampled_component = 0;
        bs.sampled_type = +BSDFFlags::GlossyReflection;

        // Ensure that this is a valid sample
        active &= neq(bs.pdf, 0.f) && Frame3f::cos_theta(bs.wo) > 0.f;

        UnpolarizedSpectrum weight;
        if (likely(m_sample_visible))
            weight = distr.smith_g1(bs.wo, m);
        else
            weight = distr.G(si.wi, bs.wo, m) * dot(si.wi, m) /
                     (cos_theta_i * Frame3f::cos_theta(m));

        // Jacobian of the half-direction mapping
        bs.pdf /= 4.f * dot(bs.wo, m);

        // Evaluate the Fresnel factor
        Complex<UnpolarizedSpectrum> eta_c(m_eta->eval(si, active),
                                           m_k->eval(si, active));

        Spectrum F;
        if constexpr (is_polarized_v<Spectrum>) {
            /* Due to lack of reciprocity in polarization-aware pBRDFs, they are
               always evaluated w.r.t. the actual light propagation direction, no
               matter the transport mode. In the following, 'wi_hat' is toward the
               light source. */
            Vector3f wi_hat = ctx.mode == TransportMode::Radiance ? bs.wo : si.wi,
                     wo_hat = ctx.mode == TransportMode::Radiance ? si.wi : bs.wo;

            // Mueller matrix for specular reflection.
            F = mueller::specular_reflection(UnpolarizedSpectrum(Frame3f::cos_theta(wi_hat)), eta_c);

            /* Apply frame reflection, according to "Stellar Polarimetry" by
               David Clarke, Appendix A.2 (A26) */
            F = mueller::reverse(F);

            /* The Stokes reference frame vector of this matrix lies in the plane
               of reflection. */
            Vector3f s_axis_in = normalize(cross(m, -wi_hat)),
                     p_axis_in = normalize(cross(-wi_hat, s_axis_in)),
                     s_axis_out = normalize(cross(m, wo_hat)),
                     p_axis_out = normalize(cross(wo_hat, s_axis_out));

            /* Rotate in/out reference vector of F s.t. it aligns with the implicit
               Stokes bases of -wi_hat & wo_hat. */
            F = mueller::rotate_mueller_basis(F,
                                              -wi_hat, p_axis_in, mueller::stokes_basis(-wi_hat),
                                               wo_hat, p_axis_out, mueller::stokes_basis(wo_hat));
        } else {
            if (likely(m_fresnel_shlick)) {
                UnpolarizedSpectrum f0 = m_f0->eval(si, active);
                F = fresnel_conductor_schlick(UnpolarizedSpectrum(dot(si.wi, m)), f0);
            } else {
                F = fresnel_conductor(UnpolarizedSpectrum(dot(si.wi, m)), eta_c);
            }
        }

        /* If requested, include the specular reflectance component */
        if (m_specular_reflectance)
            weight *= m_specular_reflectance->eval(si, active);
        return { bs, (F * weight) & active };
    }

    template <typename T>
    T wrap(const T &value, const ScalarVector2i &resolution) const {
        return clamp(value, 0, resolution - 1);
    }

    UnpolarizedSpectrum eval_envmap(Int32 lod, Point2f uv, const Wavelength &wavelengths,
                         Mask active, int32_t bottom, int32_t up) const {
        //auto& data = gather<DynamicBuffer<Float>>(m_prefiltered_envmap.data_list, lod, active);
        UnpolarizedSpectrum result(0.f);
#ifdef PREFILTERED
        for (int i = 0; i < 11; ++i) {
            auto &resolution = m_prefiltered_envmap.resolution_list[i];
            auto &data      = m_prefiltered_envmap.data_list[i];
            //Point2f uv_t = uv * Vector2f(resolution);
            //Point2u pos = min(Point2u(uv_t + 0.5), resolution - 1u);
            Point2f uv_t = uv * Vector2f(resolution - 1u);
            Point2u pos  = min(Point2u(uv_t), resolution - 2u);


            Point2f w1 = uv_t - Point2f(pos), w0 = 1.f - w1;

            const uint32_t width = resolution.x();
            UInt32 index         = pos.x() + pos.y() * width;
            Mask active_e = active;
            active_e &= eq(lod, i);
            Vector4f v00 = gather<Vector4f>(data, index, active_e),
                     v10 = gather<Vector4f>(data, index + 1, active_e),
                     v01 = gather<Vector4f>(data, index + width, active_e),
                     v11 = gather<Vector4f>(data, index + width + 1, active_e);

            ENOKI_MARK_USED(wavelengths);
            Vector4f v0 = fmadd(w0.x(), v00, w1.x() * v10),
                     v1 = fmadd(w0.x(), v01, w1.x() * v11),
                     v  = fmadd(w0.y(), v0, w1.y() * v1);

             result += head<3>(v);
        }
        result *= m_prefiltered_envmap.scale;
#else
        using Int4       = Array<Int32, 4>;
        using Int24      = Array<Int4, 2>;
        for(int i = bottom; i <= up; ++i) {
            auto &resolution = m_prefiltered_envmap.resolution_list[i];
            auto &data       = m_prefiltered_envmap.data_list[i];
            Point2f uv_t = fmadd(uv, resolution, -.5f);

            Vector2i uv_i = floor2int<Vector2i>(uv_t);

            Point2f w1 = uv_t - Point2f(uv_i), w0 = 1.f - w1;

            Int24 uv_i_w = wrap(
                Int24(Int4(0, 1, 0, 1) + uv_i.x(), Int4(0, 0, 1, 1) + uv_i.y()),
                resolution);

            Int4 index = uv_i_w.x() + uv_i_w.y() * resolution.x();
            Mask active_e = active;
            active_e &= eq(lod, i);

            Vector3f v00 = gather<Vector3f>(data, index.x(), active_e),
                     v10 = gather<Vector3f>(data, index.y(), active_e),
                     v01 = gather<Vector3f>(data, index.z(), active_e),
                     v11 = gather<Vector3f>(data, index.w(), active_e);

            Vector3f v0 = fmadd(w0.x(), v00, w1.x() * v10),
                     v1 = fmadd(w0.x(), v01, w1.x() * v11),
                     v = fmadd(w0.y(), v0, w1.y() * v1);
            result += v;
        }
#endif
        return result;
    }

    UnpolarizedSpectrum eval_envmap_multiview(Int32 lod, Point2f uv, const Wavelength &wavelengths,
                         Mask active, int32_t bottom, int32_t up, int32_t view_index) const {
        //auto& data = gather<DynamicBuffer<Float>>(m_prefiltered_envmap.data_list, lod, active);
        UnpolarizedSpectrum result(0.f);

        using Int4       = Array<Int32, 4>;
        using Int24      = Array<Int4, 2>;
        for(int i = bottom; i <= up; ++i) {
            auto &resolution = m_prefiltered_envmap.multiview_reso_list[view_index * 41 + i];
            auto &data       = m_prefiltered_envmap.multiview_data_list[view_index * 41 + i];
            Point2f uv_t = fmadd(uv, resolution, -.5f);

            Vector2i uv_i = floor2int<Vector2i>(uv_t);

            Point2f w1 = uv_t - Point2f(uv_i), w0 = 1.f - w1;

            Int24 uv_i_w = wrap(
                Int24(Int4(0, 1, 0, 1) + uv_i.x(), Int4(0, 0, 1, 1) + uv_i.y()),
                resolution);

            Int4 index = uv_i_w.x() + uv_i_w.y() * resolution.x();
            Mask active_e = active;
            active_e &= eq(lod, i);

            Vector3f v00 = gather<Vector3f>(data, index.x(), active_e),
                     v10 = gather<Vector3f>(data, index.y(), active_e),
                     v01 = gather<Vector3f>(data, index.z(), active_e),
                     v11 = gather<Vector3f>(data, index.w(), active_e);

            Vector3f v0 = fmadd(w0.x(), v00, w1.x() * v10),
                     v1 = fmadd(w0.x(), v01, w1.x() * v11),
                     v = fmadd(w0.y(), v0, w1.y() * v1);
            result += v;
        }
        return result;
    }
    Vector3f eval_brdflut(Float cos_theta, Float roughness, Mask active) const {
        using Int4 = Array<Int32, 4>;
        using Int24 = Array<Int4, 2>;
        auto &resolution = m_brdflut.resolution;
        auto &data = m_brdflut.data;
        Point2f uv = Point2f(cos_theta, roughness);
        uv = fmadd(uv, resolution, -.5f);
        
        Vector2i uv_i = floor2int<Vector2i>(uv);

        Point2f w1 = uv - Point2f(uv_i), w0 = 1.f - w1;
        
        Int24 uv_i_w = wrap(Int24(Int4(0, 1, 0, 1) + uv_i.x(),
                                          Int4(0, 0, 1, 1) + uv_i.y()), resolution);

        Int4 index = uv_i_w.x() + uv_i_w.y() * resolution.x();
        
        Vector3f v00 = gather<Vector3f>(data, index.x(), active),
                 v10 = gather<Vector3f>(data, index.y(), active),
                 v01 = gather<Vector3f>(data, index.z(), active),
                 v11 = gather<Vector3f>(data, index.w(), active);

        Vector3f v0 = fmadd(w0.x(), v00, w1.x() * v10),
                v1 = fmadd(w0.x(), v01, w1.x() * v11);
                 
        return fmadd(w0.y(), v0, w1.y() * v1);
        //return v00;
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (m_ibl) {
            // local
            Float cos_theta_i = Frame3f::cos_theta(si.wi);
            active &= cos_theta_i > 0.0f;

            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
                         none_or<false>(active)))
                return 0.f;

            //Float one(1.0f), zero(0.0f);
            //Normal3f normal = si.to_world(Normal3f(zero, zero, one));
            //Vector3f wi = si.to_world(si.wi);
            //Vector3f r = reflect(wi, normal);
#ifdef PREFILTERED
            Vector3f r = si.to_world(reflect(si.wi));
            r = m_prefiltered_envmap.world_transform->eval(si.time, active).transform_affine(r);
            
            
            Point2f uv = Point2f(atan2(r.x(), -r.z()) * math::InvTwoPi<Float>,
                                 safe_acos(r.y()) * math::InvPi<Float>);

            //Point2f uv = Point2f(atan2(r.y(), r.x()) * math::InvTwoPi<Float>,
            //                     safe_acos(r.z()) * math::InvPi<Float>);
            //Point2f uv = Point2f(atan2(r.x(), r.z()) * math::InvTwoPi<Float>,
            //                     safe_acos(r.y()) * math::InvPi<Float>);

            uv -= floor(uv);
#else
            Point2f uv = si.uv;
#endif  
            Float alpha(0.0f);
            if (likely(m_forward)) {
                alpha = m_alpha_u->eval_1(si, active);
            } else {
                // sigmoid
                Float in_alpha = m_alpha_u->eval_1(si, active);
                alpha    = Float(1.f) + exp(-in_alpha);
                alpha    = Float(1.f) / alpha;
                ////
            }
            Float roughness = safe_sqrt(alpha);
            
            //Float alpha = clamp(m_alpha_u->eval_1(si, active), Float(0.0), Float(1.0));
            //Float roughness   = sqrt(alpha);
            //Float roughness_t = roughness * Float(10);

            //Float roughness_t = roughness * Float(40);
            Float roughness_t = alpha * Float(40);
            Int32 lodf = Int32(floor(roughness_t));
            Int32 lodc = Int32(ceil(roughness_t));

            DynamicBuffer<Int32> bottom = hmin(lodf);
            DynamicBuffer<Int32> top = hmax(lodc);
            int32_t *bptr = bottom.managed().data();
            int32_t *tptr = top.managed().data();
            int bottom_value = *bptr;
            int top_value = *tptr;
            UnpolarizedSpectrum a = eval_envmap(lodf, uv, si.wavelengths, active, bottom_value, top_value);
            UnpolarizedSpectrum b = eval_envmap(lodc, uv, si.wavelengths, active, bottom_value, top_value);
            
            //UnpolarizedSpectrum a = eval_envmap(lodf, uv, si.wavelengths, active, 0, 40);
            //UnpolarizedSpectrum b = eval_envmap(lodc, uv, si.wavelengths, active, 0, 40);
            UnpolarizedSpectrum reflection = lerp(a, b, roughness_t - Float(lodf));

            Vector3f brdf = eval_brdflut(cos_theta_i, Float(1.0f) - roughness, active);

            UnpolarizedSpectrum f0(0.0f); 
            if (likely(m_forward)) {
                f0 = m_f0->eval(si, active);
            } else {
                // sigmoid
                UnpolarizedSpectrum in_f0 = m_f0->eval(si, active);
                f0 = UnpolarizedSpectrum(1.f) + exp(-in_f0);
                f0                     = Float(1.f) / f0;
                ////
            }

            UnpolarizedSpectrum f90   = m_f90->eval(si, active);
            UnpolarizedSpectrum right = f0 * brdf.x() + f90 * brdf.y();
            Spectrum res = reflection * right;

            return res & active;
            
        }

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || none_or<false>(active)))
            return 0.f;

        // Calculate the half-direction vector
        Vector3f H = normalize(wo + si.wi);

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type,
                                     m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        // Evaluate the microfacet normal distribution
        Float D = distr.eval(H);

        active &= neq(D, 0.f);

        // Evaluate Smith's shadow-masking function
        Float G = distr.G(si.wi, wo, H);

        // Evaluate the full microfacet model (except Fresnel)
        UnpolarizedSpectrum result = D * G / (4.f * Frame3f::cos_theta(si.wi));

        // Evaluate the Fresnel factor
        Complex<UnpolarizedSpectrum> eta_c(m_eta->eval(si, active),
                                           m_k->eval(si, active));

        Spectrum F;
        if constexpr (is_polarized_v<Spectrum>) {
            /* Due to lack of reciprocity in polarization-aware pBRDFs, they are
               always evaluated w.r.t. the actual light propagation direction, no
               matter the transport mode. In the following, 'wi_hat' is toward the
               light source. */
            Vector3f wi_hat = ctx.mode == TransportMode::Radiance ? wo : si.wi,
                     wo_hat = ctx.mode == TransportMode::Radiance ? si.wi : wo;

            // Mueller matrix for specular reflection.
            F = mueller::specular_reflection(UnpolarizedSpectrum(Frame3f::cos_theta(wi_hat)), eta_c);

            /* Apply frame reflection, according to "Stellar Polarimetry" by
               David Clarke, Appendix A.2 (A26) */
            F = mueller::reverse(F);

            /* The Stokes reference frame vector of this matrix lies in the plane
               of reflection. */
            Vector3f s_axis_in  = normalize(cross(H, -wi_hat)),
                     p_axis_in  = normalize(cross(-wi_hat, s_axis_in)),
                     s_axis_out = normalize(cross(H, wo_hat)),
                     p_axis_out = normalize(cross(wo_hat, s_axis_out));

            /* Rotate in/out reference vector of F s.t. it aligns with the implicit
               Stokes bases of -wi_hat & wo_hat. */
            F = mueller::rotate_mueller_basis(F,
                                              -wi_hat, p_axis_in, mueller::stokes_basis(-wi_hat),
                                               wo_hat, p_axis_out, mueller::stokes_basis(wo_hat));
        } else {
            if (likely(m_fresnel_shlick)) {
                UnpolarizedSpectrum f0 = m_f0->eval(si, active);
                F = fresnel_conductor_schlick(UnpolarizedSpectrum(dot(si.wi, H)), f0);
            } else {
                F = fresnel_conductor(UnpolarizedSpectrum(dot(si.wi, H)), eta_c);
            }
        }

        /* If requested, include the specular reflectance component */
        if (m_specular_reflectance)
            result *= m_specular_reflectance->eval(si, active);
        return (F * result) & active;
    }

    Spectrum eval_multiview(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, const int view_index_, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (m_ibl && m_ibl_multiview) {
            if (view_index_ < 0) {
                return 0.f;
            }
            int view_index = view_index_;

            view_index %= m_prefiltered_envmap.view_count;
            // local
            Float cos_theta_i = Frame3f::cos_theta(si.wi);
            active &= cos_theta_i > 0.0f;

            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
                         none_or<false>(active)))
                return 0.f;
            //Float one(1.0f), zero(0.0f);
            //Normal3f normal = si.to_world(Normal3f(zero, zero, one));
            //Vector3f wi = si.to_world(si.wi);
            //Vector3f r = reflect(wi, normal);
#ifdef PREFILTERED
            Vector3f r = si.to_world(reflect(si.wi));
            r = m_prefiltered_envmap.world_transform->eval(si.time, active).transform_affine(r);
            
            
            Point2f uv = Point2f(atan2(r.x(), -r.z()) * math::InvTwoPi<Float>,
                                 safe_acos(r.y()) * math::InvPi<Float>);

            //Point2f uv = Point2f(atan2(r.y(), r.x()) * math::InvTwoPi<Float>,
            //                     safe_acos(r.z()) * math::InvPi<Float>);
            //Point2f uv = Point2f(atan2(r.x(), r.z()) * math::InvTwoPi<Float>,
            //                     safe_acos(r.y()) * math::InvPi<Float>);

            uv -= floor(uv);
#else
            Point2f uv = si.uv;
#endif  
            Float alpha(0.0f);
            if (likely(m_forward)) {
                alpha = m_alpha_u->eval_1(si, active);
            } else {
                // sigmoid
                Float in_alpha = m_alpha_u->eval_1(si, active);
                alpha    = Float(1.f) + exp(-in_alpha);
                alpha    = Float(1.f) / alpha;
                ////
            }
            Float roughness = safe_sqrt(alpha);
            
            //Float alpha = clamp(m_alpha_u->eval_1(si, active), Float(0.0), Float(1.0));
            //Float roughness   = sqrt(alpha);
            //Float roughness_t = roughness * Float(10);
            
            //Float roughness_t = roughness * Float(40);
            Float roughness_t = alpha * Float(40);
            Int32 lodf = Int32(floor(roughness_t));
            Int32 lodc = Int32(ceil(roughness_t));

            DynamicBuffer<Int32> bottom = hmin(lodf);
            DynamicBuffer<Int32> top = hmax(lodc);
            int32_t *bptr = bottom.managed().data();
            int32_t *tptr = top.managed().data();
            int bottom_value = *bptr;
            int top_value = *tptr;
            UnpolarizedSpectrum a = eval_envmap_multiview(lodf, uv, si.wavelengths, active, bottom_value, top_value, view_index);
            UnpolarizedSpectrum b = eval_envmap_multiview(lodc, uv, si.wavelengths, active, bottom_value, top_value, view_index);
            
            //UnpolarizedSpectrum a = eval_envmap(lodf, uv, si.wavelengths, active, 0, 40);
            //UnpolarizedSpectrum b = eval_envmap(lodc, uv, si.wavelengths, active, 0, 40);
            UnpolarizedSpectrum reflection = lerp(a, b, roughness_t - Float(lodf));

            Vector3f brdf = eval_brdflut(cos_theta_i, Float(1.0f) - roughness, active);

            UnpolarizedSpectrum f0(0.0f); 
            if (likely(m_forward)) {
                f0 = m_f0->eval(si, active);
            } else {
                // sigmoid
                UnpolarizedSpectrum in_f0 = m_f0->eval(si, active);
                f0 = UnpolarizedSpectrum(1.f) + exp(-in_f0);
                f0                     = Float(1.f) / f0;
                ////
            }

            UnpolarizedSpectrum f90   = m_f90->eval(si, active);
            UnpolarizedSpectrum right = f0 * brdf.x() + f90 * brdf.y();
            Spectrum res = reflection * right;

            return res & active;
            
        }

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || none_or<false>(active)))
            return 0.f;

        // Calculate the half-direction vector
        Vector3f H = normalize(wo + si.wi);

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type,
                                     m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        // Evaluate the microfacet normal distribution
        Float D = distr.eval(H);

        active &= neq(D, 0.f);

        // Evaluate Smith's shadow-masking function
        Float G = distr.G(si.wi, wo, H);

        // Evaluate the full microfacet model (except Fresnel)
        UnpolarizedSpectrum result = D * G / (4.f * Frame3f::cos_theta(si.wi));

        // Evaluate the Fresnel factor
        Complex<UnpolarizedSpectrum> eta_c(m_eta->eval(si, active),
                                           m_k->eval(si, active));

        Spectrum F;
        if constexpr (is_polarized_v<Spectrum>) {
            /* Due to lack of reciprocity in polarization-aware pBRDFs, they are
               always evaluated w.r.t. the actual light propagation direction, no
               matter the transport mode. In the following, 'wi_hat' is toward the
               light source. */
            Vector3f wi_hat = ctx.mode == TransportMode::Radiance ? wo : si.wi,
                     wo_hat = ctx.mode == TransportMode::Radiance ? si.wi : wo;

            // Mueller matrix for specular reflection.
            F = mueller::specular_reflection(UnpolarizedSpectrum(Frame3f::cos_theta(wi_hat)), eta_c);

            /* Apply frame reflection, according to "Stellar Polarimetry" by
               David Clarke, Appendix A.2 (A26) */
            F = mueller::reverse(F);

            /* The Stokes reference frame vector of this matrix lies in the plane
               of reflection. */
            Vector3f s_axis_in  = normalize(cross(H, -wi_hat)),
                     p_axis_in  = normalize(cross(-wi_hat, s_axis_in)),
                     s_axis_out = normalize(cross(H, wo_hat)),
                     p_axis_out = normalize(cross(wo_hat, s_axis_out));

            /* Rotate in/out reference vector of F s.t. it aligns with the implicit
               Stokes bases of -wi_hat & wo_hat. */
            F = mueller::rotate_mueller_basis(F,
                                              -wi_hat, p_axis_in, mueller::stokes_basis(-wi_hat),
                                               wo_hat, p_axis_out, mueller::stokes_basis(wo_hat));
        } else {
            if (likely(m_fresnel_shlick)) {
                UnpolarizedSpectrum f0 = m_f0->eval(si, active);
                F = fresnel_conductor_schlick(UnpolarizedSpectrum(dot(si.wi, H)), f0);
            } else {
                F = fresnel_conductor(UnpolarizedSpectrum(dot(si.wi, H)), eta_c);
            }
        }

        /* If requested, include the specular reflectance component */
        if (m_specular_reflectance)
            result *= m_specular_reflectance->eval(si, active);
        return (F * result) & active;
    }
    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        // Calculate the half-direction vector
        Vector3f m = normalize(wo + si.wi);
        
        /* Filter cases where the micro/macro-surface don't agree on the side.
           This logic is evaluated in smith_g1() called as part of the eval()
           and sample() methods and needs to be replicated in the probability
           density computation as well. */
        active &= cos_theta_i   > 0.f && cos_theta_o   > 0.f &&
                 dot(si.wi, m) > 0.f && dot(wo,    m) > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || none_or<false>(active)))
            return 0.f;

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type,
                                     m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        Float result;
        if (likely(m_sample_visible))
            result = distr.eval(m) * distr.smith_g1(si.wi, m) /
                     (4.f * cos_theta_i);
        else
            result = distr.pdf(si.wi, m) / (4.f * dot(wo, m));

        return select(active, result, 0.f);
    }

    Spectrum eval_window(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const SurfaceInteraction3f &sub_si, const Vector3f &wo,
                         Mask active, Mask sub_active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, sub_active);

        if (m_ibl) {
            // local
            Float cos_theta_i = Frame3f::cos_theta(sub_si.wi);
            sub_active &= cos_theta_i > 0.0f;

            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
                         none_or<false>(sub_active)))
                return 0.f;

            // Float one(1.0f), zero(0.0f);
            // Normal3f normal = si.to_world(Normal3f(zero, zero, one));
            // Vector3f wi = si.to_world(si.wi);
            // Vector3f r = reflect(wi, normal);
#ifdef PREFILTERED
            Vector3f r = si.to_world(reflect(si.wi));
            r = m_prefiltered_envmap.world_transform->eval(si.time, active)
                    .transform_affine(r);

            Point2f uv = Point2f(atan2(r.x(), -r.z()) * math::InvTwoPi<Float>,
                                 safe_acos(r.y()) * math::InvPi<Float>);

            // Point2f uv = Point2f(atan2(r.y(), r.x()) * math::InvTwoPi<Float>,
            //                     safe_acos(r.z()) * math::InvPi<Float>);
            // Point2f uv = Point2f(atan2(r.x(), r.z()) * math::InvTwoPi<Float>,
            //                     safe_acos(r.y()) * math::InvPi<Float>);

            uv -= floor(uv);
#else
            Point2f uv = sub_si.uv;
#endif
            // sigmoid
            Float in_alpha = m_alpha_u->eval_1(si, sub_active);
            Float alpha    = Float(1.f) + exp(-in_alpha);
            alpha          = Float(1.f) / alpha;
            ////
            Float roughness   = safe_sqrt(alpha);

            //Float alpha = m_alpha_u->eval_1(si, active);
            //Float roughness   = safe_sqrt(alpha);

            // Float roughness_t = roughness * Float(10);
            Float roughness_t = roughness * Float(40);
            Int32 lodf        = Int32(floor(roughness_t));
            Int32 lodc        = Int32(ceil(roughness_t));

            DynamicBuffer<Int32> bottom = hmin(lodf);
            DynamicBuffer<Int32> top = hmax(lodc);
            int32_t *bptr = bottom.managed().data();
            int32_t *tptr = top.managed().data();
            int bottom_value = *bptr;
            int top_value = *tptr;
            UnpolarizedSpectrum a = eval_envmap(lodf, uv, si.wavelengths, sub_active, bottom_value, top_value);
            UnpolarizedSpectrum b = eval_envmap(lodc, uv, si.wavelengths, sub_active, bottom_value, top_value);

            //UnpolarizedSpectrum a = eval_envmap(lodf, uv, si.wavelengths, sub_active, 0, 40);
            //UnpolarizedSpectrum b = eval_envmap(lodc, uv, si.wavelengths, sub_active, 0, 40);

            UnpolarizedSpectrum reflection = lerp(a, b, roughness_t - Float(lodf));

            Vector3f brdf = eval_brdflut(cos_theta_i, Float(1.0f) - roughness, sub_active);

            // sigmoid
            UnpolarizedSpectrum in_f0 = m_f0->eval(si, sub_active);
            UnpolarizedSpectrum f0    = UnpolarizedSpectrum(1.f) + exp(-in_f0);
            f0                        = UnpolarizedSpectrum(1.f) / f0;
            ////

            UnpolarizedSpectrum right = f0 * brdf.x() + brdf.y();
            Spectrum res              = reflection * right;

            return res & sub_active;
        }

        Float cos_theta_i = Frame3f::cos_theta(sub_si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        sub_active &= cos_theta_i > 0.f && cos_theta_o > 0.f;
        //active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
                     none_or<false>(sub_active)))
            return 0.f;

        // Calculate the half-direction vector
        Vector3f H = normalize(wo + sub_si.wi);

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type, m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        // Evaluate the microfacet normal distribution
        Float D = distr.eval(H);

        sub_active &= neq(D, 0.f);

        // Evaluate Smith's shadow-masking function
        Float G = distr.G(sub_si.wi, wo, H);

        // Evaluate the full microfacet model (except Fresnel)
        UnpolarizedSpectrum result = D * G / (4.f * Frame3f::cos_theta(sub_si.wi));

        // Evaluate the Fresnel factor
        Complex<UnpolarizedSpectrum> eta_c(m_eta->eval(si, active),
                                           m_k->eval(si, active));

        Spectrum F;
        if constexpr (is_polarized_v<Spectrum>) {
            /* Due to lack of reciprocity in polarization-aware pBRDFs, they are
               always evaluated w.r.t. the actual light propagation direction,
               no matter the transport mode. In the following, 'wi_hat' is
               toward the light source. */
            Vector3f wi_hat = ctx.mode == TransportMode::Radiance ? wo : sub_si.wi,
                     wo_hat = ctx.mode == TransportMode::Radiance ? sub_si.wi : wo;

            // Mueller matrix for specular reflection.
            F = mueller::specular_reflection(
                UnpolarizedSpectrum(Frame3f::cos_theta(wi_hat)), eta_c);

            /* Apply frame reflection, according to "Stellar Polarimetry" by
               David Clarke, Appendix A.2 (A26) */
            F = mueller::reverse(F);

            /* The Stokes reference frame vector of this matrix lies in the
               plane of reflection. */
            Vector3f s_axis_in  = normalize(cross(H, -wi_hat)),
                     p_axis_in  = normalize(cross(-wi_hat, s_axis_in)),
                     s_axis_out = normalize(cross(H, wo_hat)),
                     p_axis_out = normalize(cross(wo_hat, s_axis_out));

            /* Rotate in/out reference vector of F s.t. it aligns with the
               implicit Stokes bases of -wi_hat & wo_hat. */
            F = mueller::rotate_mueller_basis(
                F, -wi_hat, p_axis_in, mueller::stokes_basis(-wi_hat), wo_hat,
                p_axis_out, mueller::stokes_basis(wo_hat));
        } else {
            if (likely(m_fresnel_shlick)) {
                UnpolarizedSpectrum f0 = m_f0->eval(si, sub_active);
                F                      = fresnel_conductor_schlick(
                    UnpolarizedSpectrum(dot(sub_si.wi, H)), f0);
            } else {
                F = fresnel_conductor(UnpolarizedSpectrum(dot(sub_si.wi, H)),
                                      eta_c);
            }
        }

        /* If requested, include the specular reflectance component */
        if (m_specular_reflectance)
            result *= m_specular_reflectance->eval(si, active);
        return (F * result) & sub_active;
    }

    Float pdf_window(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                     const SurfaceInteraction3f &sub_si, const Vector3f &wo,
                     Mask active, Mask sub_active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, sub_active);

        Float cos_theta_i = Frame3f::cos_theta(sub_si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        // Calculate the half-direction vector
        Vector3f m = normalize(wo + sub_si.wi);

        /* Filter cases where the micro/macro-surface don't agree on the side.
           This logic is evaluated in smith_g1() called as part of the eval()
           and sample() methods and needs to be replicated in the probability
           density computation as well. */
        sub_active &= cos_theta_i > 0.f && cos_theta_o > 0.f &&
                  dot(sub_si.wi, m) > 0.f && dot(wo, m) > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
                     none_or<false>(sub_active)))
            return 0.f;

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type, m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        Float result;
        if (likely(m_sample_visible))
            result =
                distr.eval(m) * distr.smith_g1(sub_si.wi, m) / (4.f * cos_theta_i);
        else
            result = distr.pdf(sub_si.wi, m) / (4.f * dot(wo, m));

        return select(sub_active, result, 0.f);
    }

    std::pair<BSDFSample3f, Spectrum> sample_window(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             const SurfaceInteraction3f &sub_si,
                                             Float /* sample1 */,
                                             const Point2f &sample2,
                                             Mask active,
                                             Mask sub_active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, sub_active);

        BSDFSample3f bs   = zero<BSDFSample3f>();
        Float cos_theta_i = Frame3f::cos_theta(sub_si.wi);
        sub_active &= cos_theta_i > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
                     none_or<false>(sub_active)))
            return { bs, 0.f };

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type, m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        // Sample M, the microfacet normal
        Normal3f m;
        std::tie(m, bs.pdf) = distr.sample(sub_si.wi, sample2);

        // Perfect specular reflection based on the microfacet normal
        bs.wo                = reflect(sub_si.wi, m);
        bs.eta               = 1.f;
        bs.sampled_component = 0;
        bs.sampled_type      = +BSDFFlags::GlossyReflection;

        // Ensure that this is a valid sample
        sub_active &= neq(bs.pdf, 0.f) && Frame3f::cos_theta(bs.wo) > 0.f;

        UnpolarizedSpectrum weight;
        if (likely(m_sample_visible))
            weight = distr.smith_g1(bs.wo, m);
        else
            weight = distr.G(sub_si.wi, bs.wo, m) * dot(sub_si.wi, m) /
                     (cos_theta_i * Frame3f::cos_theta(m));

        // Jacobian of the half-direction mapping
        bs.pdf /= 4.f * dot(bs.wo, m);

        // Evaluate the Fresnel factor
        Complex<UnpolarizedSpectrum> eta_c(m_eta->eval(si, active),
                                           m_k->eval(si, active));

        Spectrum F;
        if constexpr (is_polarized_v<Spectrum>) {
            /* Due to lack of reciprocity in polarization-aware pBRDFs, they are
               always evaluated w.r.t. the actual light propagation direction,
               no matter the transport mode. In the following, 'wi_hat' is
               toward the light source. */
            Vector3f wi_hat =
                         ctx.mode == TransportMode::Radiance ? bs.wo : sub_si.wi,
                     wo_hat =
                         ctx.mode == TransportMode::Radiance ? sub_si.wi : bs.wo;

            // Mueller matrix for specular reflection.
            F = mueller::specular_reflection(
                UnpolarizedSpectrum(Frame3f::cos_theta(wi_hat)), eta_c);

            /* Apply frame reflection, according to "Stellar Polarimetry" by
               David Clarke, Appendix A.2 (A26) */
            F = mueller::reverse(F);

            /* The Stokes reference frame vector of this matrix lies in the
               plane of reflection. */
            Vector3f s_axis_in  = normalize(cross(m, -wi_hat)),
                     p_axis_in  = normalize(cross(-wi_hat, s_axis_in)),
                     s_axis_out = normalize(cross(m, wo_hat)),
                     p_axis_out = normalize(cross(wo_hat, s_axis_out));

            /* Rotate in/out reference vector of F s.t. it aligns with the
               implicit Stokes bases of -wi_hat & wo_hat. */
            F = mueller::rotate_mueller_basis(
                F, -wi_hat, p_axis_in, mueller::stokes_basis(-wi_hat), wo_hat,
                p_axis_out, mueller::stokes_basis(wo_hat));
        } else {
            if (likely(m_fresnel_shlick)) {
                UnpolarizedSpectrum f0 = m_f0->eval(si, active);
                F                      = fresnel_conductor_schlick(
                    UnpolarizedSpectrum(dot(sub_si.wi, m)), f0);
            } else {
                F = fresnel_conductor(UnpolarizedSpectrum(dot(sub_si.wi, m)),
                                      eta_c);
            }
        }

        /* If requested, include the specular reflectance component */
        if (m_specular_reflectance)
            weight *= m_specular_reflectance->eval(si, active);
        return { bs, (F * weight) & sub_active };
    }

    void catch_irradiance(const BSDFContext &ctx,
        const SurfaceInteraction3f &si, 
        const Vector3f &wo,
        const Spectrum &emitter_val,
        Mask active) const override {

        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);


        if (!m_catch_irradiance) {
            return;
        }

        Spectrum irradiance_val = select(active, emitter_val, 0.f);

        m_catch_bitmap->catch_spec(irradiance_val, si, active);
    }

    void acc_irradiance(float factor) const override {
        if (!m_catch_irradiance || factor <= 0) {
            return;
        }
        
        auto m_data  = m_catch_bitmap->m_data.managed();
        auto m_weights  = m_catch_bitmap->m_weights.managed();
        const ScalarFloat *fptr = m_data.data();
        const ScalarFloat *wptr = m_weights.data();
        float *cdata            = new float[hprod(m_catch_bitmap->m_resolution) * 3];

        //factor *= m_envmap_scale;
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
        //std::cout << "aha! save glossy irradiance\n";
    }

    void save_irradiance() const override {
        if (!m_catch_irradiance || m_catch_bitmap == nullptr || m_catch_bitmap->m_bitmap == nullptr) {
            return;
        }
        m_catch_bitmap->m_bitmap->write(fs::path(m_irradiance_filename));
        delete m_catch_bitmap;
    }

    void traverse(TraversalCallback *callback) override {
        if (!has_flag(m_flags, BSDFFlags::Anisotropic))
            callback->put_object("alpha", m_alpha_u.get());
        else {
            callback->put_object("alpha_u", m_alpha_u.get());
            callback->put_object("alpha_v", m_alpha_v.get());
        }
        callback->put_object("eta", m_eta.get());
        callback->put_object("k", m_k.get());
        callback->put_object("f0", m_f0.get());
        callback->put_object("f90", m_f90.get());
        if (m_specular_reflectance)
            callback->put_object("specular_reflectance", m_specular_reflectance.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "RoughConductor[" << std::endl
            << "  distribution = " << m_type << "," << std::endl
            << "  sample_visible = " << m_sample_visible << "," << std::endl
            << "  alpha_u = " << string::indent(m_alpha_u) << "," << std::endl
            << "  alpha_v = " << string::indent(m_alpha_v) << "," << std::endl;
        if (m_specular_reflectance)
           oss << "  specular_reflectance = " << string::indent(m_specular_reflectance) << "," << std::endl;
        oss << "  eta = " << string::indent(m_eta) << "," << std::endl
            << "  k = " << string::indent(m_k) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    /// Specifies the type of microfacet distribution
    MicrofacetType m_type;
    /// Anisotropic roughness values
    ref<Texture> m_alpha_u, m_alpha_v;
    /// Importance sample the distribution of visible normals?
    bool m_sample_visible;
    /// Relative refractive index (real component)
    ref<Texture> m_eta;
    /// Relative refractive index (imaginary component).
    ref<Texture> m_k;
    /// Specular reflectance component
    ref<Texture> m_specular_reflectance;

    // use shlick
    bool m_fresnel_shlick = false;
    // 'zero angle' reflentance value
    ref<Texture> m_f0;
    ref<Texture> m_f90;

    // use ibl
    bool m_ibl = false;
    bool m_ibl_multiview = false;
    struct {
        DynamicBuffer<Float> data;
        ScalarVector2i resolution;
    } m_brdflut;
    struct {
        const int view_count = 4;
        Vector<DynamicBuffer<Float>, 164> multiview_data_list;
        Vector<ScalarVector2u, 164> multiview_reso_list;

        Vector<DynamicBuffer<Float>, 41> data_list;
        Vector<ScalarVector2u, 41> resolution_list;
        ref<const AnimatedTransform> world_transform;
        float scale;
    } m_prefiltered_envmap;

    // catch irradiance
    bool m_catch_irradiance = false;
    int m_irradiance_width, m_irradiance_height;
    std::string m_irradiance_filename;
    float m_envmap_scale;
    CatchBitmap *m_catch_bitmap;

    // forward rendering
    bool m_forward = false;
};

MTS_IMPLEMENT_CLASS_VARIANT(RoughConductor, BSDF)
MTS_EXPORT_PLUGIN(RoughConductor, "Rough conductor")
NAMESPACE_END(mitsuba)