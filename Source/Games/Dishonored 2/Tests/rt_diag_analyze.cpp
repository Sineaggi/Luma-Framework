// RT diagnostic capture analysis (see RTAO_NOTES.md "Diagnostic capture", main.cpp g_rtdiag).
// Usage: rt_diag_analyze <capture folder>
// Build (x64 VS prompt): cl /nologo /std:c++20 /EHsc /O2 rt_diag_analyze.cpp
//
// For every pair of consecutive captures in the same phase, each pixel of the newer capture is reprojected into the
// older one through its depth and the two cameras (so turning the camera is not counted as change). Where both
// captures saw the same surface (positions within 2 % of the distance), the stages are compared:
//   game_shadow, rt_raw, rt_denoised: "flipped" if the shadow value changed by more than 0.35 (lit <-> shadowed)
//   scene: the ray tracing scene's color class (capture source, or nothing) changed (a caster appeared/vanished there)
//   final: luminance changed by more than 0.15
// A stage that flips while the stages before it don't is where the change starts. The game's own shadow is the
// reference: it doesn't pop in game.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Capture
{
   int index = -1;
   std::map<std::string, std::string> meta;
   double vp[16] = {}, inv_vp[16] = {};
   double cam[3] = {}, fwd[3] = {};
   int phase = 0;
   uint32_t w = 0, h = 0;
   std::vector<float> depth;
   std::map<std::string, std::vector<uint8_t>> gray; // game_shadow, rt_raw, rt_denoised
   std::vector<uint8_t> scene, final_image, scene_sun; // RGB
   struct Probe { uint32_t key, flags; double pos[3]; };
   std::vector<Probe> probes;
};

static bool Invert(const double m[16], double out[16])
{
   double inv[16];
   inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
   inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
   inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
   inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
   inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
   inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
   inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
   inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
   inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
   inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
   inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
   inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
   inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
   inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
   inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
   inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
   double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
   if (std::fabs(det) < 1e-30)
      return false;
   for (int i = 0; i < 16; i++)
      out[i] = inv[i] / det;
   return true;
}

// Row vector convention (the game's): out = [x y z w] * M
static void Mul(const double v[4], const double m[16], double out[4])
{
   for (int c = 0; c < 4; c++)
      out[c] = v[0] * m[c] + v[1] * m[4 + c] + v[2] * m[8 + c] + v[3] * m[12 + c];
}

static std::vector<uint8_t> ReadBytes(const fs::path& path)
{
   std::ifstream file(path, std::ios::binary);
   return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static bool ParseList(const std::string& s, double* out, int count)
{
   std::stringstream ss(s);
   std::string item;
   int i = 0;
   while (i < count && std::getline(ss, item, ','))
      out[i++] = std::stod(item);
   return i == count;
}

static bool Load(const fs::path& dir, int index, Capture& c)
{
   char name[32];
   snprintf(name, sizeof(name), "c%04d", index);
   std::ifstream meta(dir / (std::string(name) + ".txt"));
   if (!meta)
      return false;
   std::string line;
   while (std::getline(meta, line))
   {
      const size_t eq = line.find('=');
      if (eq == std::string::npos)
         continue;
      if (line.substr(0, eq) == "probe")
      {
         Capture::Probe probe = {};
         unsigned key = 0, flags = 0;
         if (sscanf(line.c_str() + eq + 1, "%X,%u,%lf,%lf,%lf", &key, &flags, &probe.pos[0], &probe.pos[1], &probe.pos[2]) == 5)
         {
            probe.key = key;
            probe.flags = flags;
            c.probes.push_back(probe);
         }
         continue;
      }
      c.meta[line.substr(0, eq)] = line.substr(eq + 1);
   }
   c.index = index;
   if (!ParseList(c.meta["view_proj"], c.vp, 16) || !Invert(c.vp, c.inv_vp))
      return false;
   ParseList(c.meta["cam_pos"], c.cam, 3);
   ParseList(c.meta["cam_fwd"], c.fwd, 3);
   c.phase = std::stoi(c.meta["phase"]);
   double size[2] = {};
   if (!ParseList(c.meta["size_depth"], size, 2))
      return false;
   c.w = uint32_t(size[0]);
   c.h = uint32_t(size[1]);
   const size_t n = size_t(c.w) * c.h;
   std::vector<uint8_t> d = ReadBytes(dir / (std::string(name) + "_depth.bin"));
   if (d.size() != n * 4)
      return false;
   c.depth.resize(n);
   std::memcpy(c.depth.data(), d.data(), d.size());
   for (const char* g : { "game_shadow", "rt_raw", "rt_denoised" })
   {
      std::vector<uint8_t> v = ReadBytes(dir / (std::string(name) + "_" + g + ".bin"));
      if (v.size() == n)
         c.gray[g] = std::move(v);
   }
   std::vector<uint8_t> s = ReadBytes(dir / (std::string(name) + "_scene.bin"));
   if (s.size() == n * 3)
      c.scene = std::move(s);
   std::vector<uint8_t> ss = ReadBytes(dir / (std::string(name) + "_scene_sun.bin"));
   if (ss.size() == n * 3)
      c.scene_sun = std::move(ss);
   std::vector<uint8_t> f = ReadBytes(dir / (std::string(name) + "_final.bin"));
   if (f.size() == n * 3)
      c.final_image = std::move(f);
   return true;
}

static bool World(const Capture& c, uint32_t x, uint32_t y, double out[3])
{
   const float d = c.depth[size_t(y) * c.w + x];
   if (!(d > 0.f) || d >= 1.f) // Reversed Z: 0 is the sky
      return false;
   const double ndc[4] = { (x + 0.5) / c.w * 2.0 - 1.0, 1.0 - (y + 0.5) / c.h * 2.0, d, 1.0 };
   double h[4];
   Mul(ndc, c.inv_vp, h);
   if (std::fabs(h[3]) < 1e-12)
      return false;
   for (int i = 0; i < 3; i++)
      out[i] = h[i] / h[3];
   return true;
}

// The capture source class of a scene view color: grey (nothing hit) or the channel ratios (the per triangle shade and
// depth falloff scale all channels alike)
static int SceneClass(const uint8_t* rgb)
{
   const int m = std::max<int>(rgb[0], std::max<int>(rgb[1], rgb[2]));
   if (m < 45 && std::abs(rgb[0] - rgb[1]) < 8 && std::abs(rgb[1] - rgb[2]) < 8)
      return 0;
   int cls = 1;
   for (int c = 0; c < 3; c++)
      cls = cls * 8 + int(std::lround(rgb[c] * 4.0 / std::max(m, 1)));
   return cls;
}

static float Luma(const uint8_t* rgb)
{
   return (0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2]) / 255.f;
}

int main(int argc, char** argv)
{
   if (argc < 2)
   {
      printf("usage: rt_diag_analyze <capture folder>\n");
      return 2;
   }
   const fs::path dir = argv[1];
   std::vector<Capture> captures;
   for (int i = 0; i < 10000; i++)
   {
      Capture c;
      if (!Load(dir, i, c))
      {
         if (!fs::exists(dir / (std::string("c") + (i < 10 ? "000" : i < 100 ? "00" : i < 1000 ? "0" : "") + std::to_string(i) + ".txt")))
            break;
         printf("capture %d: incomplete, skipped\n", i);
         continue;
      }
      captures.push_back(std::move(c));
   }
   printf("%zu captures in %s\n", captures.size(), dir.string().c_str());
   if (captures.size() < 2)
      return 1;
   const Capture& first = captures.front();
   printf("stages: game_shadow %s, rt_raw %s, rt_denoised %s, scene %s, final %s; %ux%u\n", first.gray.count("game_shadow") ? "yes" : "NO", first.gray.count("rt_raw") ? "yes" : "NO",
      first.gray.count("rt_denoised") ? "yes" : "NO", first.scene.empty() ? "NO" : "yes", first.final_image.empty() ? "NO" : "yes", first.w, first.h);
   printf("\n  pair       phase  move(m) turn(deg) compared |  flipped %%: game_sh  rt_raw  rt_den   scene   final | rt_den flips where scene same | lighting used RT/game | TLAS  new evict\n");

   struct Sum
   {
      double flips[5] = {};
      double rt_same_scene = 0;
      int pairs = 0;
   } sums[2];
   for (size_t k = 1; k < captures.size(); k++)
   {
      const Capture& a = captures[k - 1];
      const Capture& b = captures[k];
      if (a.phase != b.phase || a.w != b.w || a.h != b.h)
         continue;
      const double move = std::sqrt((a.cam[0] - b.cam[0]) * (a.cam[0] - b.cam[0]) + (a.cam[1] - b.cam[1]) * (a.cam[1] - b.cam[1]) + (a.cam[2] - b.cam[2]) * (a.cam[2] - b.cam[2]));
      const double dot = std::clamp(a.fwd[0] * b.fwd[0] + a.fwd[1] * b.fwd[1] + a.fwd[2] * b.fwd[2], -1.0, 1.0);
      const double turn = std::acos(dot) * 57.29578;
      uint64_t compared = 0, flips[5] = {}, scene_same = 0, rt_flip_scene_same = 0;
      for (uint32_t y = 0; y < b.h; y++)
         for (uint32_t x = 0; x < b.w; x++)
         {
            double p[3];
            if (!World(b, x, y, p))
               continue;
            const double hp[4] = { p[0], p[1], p[2], 1.0 };
            double clip[4];
            Mul(hp, a.vp, clip);
            if (clip[3] <= 1e-6)
               continue;
            const double u = (clip[0] / clip[3] * 0.5 + 0.5) * a.w, v = (0.5 - clip[1] / clip[3] * 0.5) * a.h;
            if (u < 0 || v < 0 || u >= a.w || v >= a.h)
               continue;
            const uint32_t ax = uint32_t(u), ay = uint32_t(v);
            double q[3];
            if (!World(a, ax, ay, q))
               continue;
            const double dist = std::sqrt((p[0] - b.cam[0]) * (p[0] - b.cam[0]) + (p[1] - b.cam[1]) * (p[1] - b.cam[1]) + (p[2] - b.cam[2]) * (p[2] - b.cam[2]));
            const double gap = std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
            if (gap > 0.02 * dist + 0.02)
               continue; // Not the same surface (disocclusion)
            compared++;
            const size_t ib = size_t(y) * b.w + x, ia = size_t(ay) * a.w + ax;
            const char* gray_stages[3] = { "game_shadow", "rt_raw", "rt_denoised" };
            bool rt_flip = false;
            for (int s = 0; s < 3; s++)
            {
               auto sa = a.gray.find(gray_stages[s]), sb = b.gray.find(gray_stages[s]);
               if (sa == a.gray.end() || sb == b.gray.end())
                  continue;
               const bool flip = std::abs(int(sb->second[ib]) - int(sa->second[ia])) > 89; // 0.35
               flips[s] += flip;
               if (s == 2)
                  rt_flip = flip;
            }
            bool same_scene = true;
            if (!a.scene.empty() && !b.scene.empty())
            {
               same_scene = SceneClass(&a.scene[ia * 3]) == SceneClass(&b.scene[ib * 3]);
               flips[3] += !same_scene;
            }
            if (same_scene)
            {
               scene_same++;
               rt_flip_scene_same += rt_flip;
            }
            if (!a.final_image.empty() && !b.final_image.empty())
               flips[4] += std::fabs(Luma(&a.final_image[ia * 3]) - Luma(&b.final_image[ib * 3])) > 0.15f;
         }
      if (!compared)
         continue;
      double pct[5];
      for (int s = 0; s < 5; s++)
         pct[s] = 100.0 * flips[s] / compared;
      const double rt_same = scene_same ? 100.0 * rt_flip_scene_same / scene_same : 0.0;
      // The sun view (orthographic around the camera, so only comparable as is while the camera barely moves)
      char sun_text[16] = "-";
      if (!a.scene_sun.empty() && !b.scene_sun.empty() && move < 0.05)
      {
         uint64_t changed = 0;
         for (size_t i = 0; i < size_t(b.w) * b.h; i++)
            changed += SceneClass(&a.scene_sun[i * 3]) != SceneClass(&b.scene_sun[i * 3]);
         snprintf(sun_text, sizeof(sun_text), "%.2f%%", 100.0 * changed / (double(b.w) * b.h));
      }
      Sum& sum = sums[b.phase & 1];
      for (int s = 0; s < 5; s++)
         sum.flips[s] += pct[s];
      sum.rt_same_scene += rt_same;
      sum.pairs++;
      auto m = [&](const Capture& c, const char* key) { auto it = c.meta.find(key); return it != c.meta.end() ? it->second : std::string("?"); };
      printf("  c%04d-c%04d   %d    %6.2f   %6.1f  %7llu | %15.2f %7.2f %7.2f %7.2f %7.2f | %29.2f | %10s/%-4s | %5s %4s %5s\n", a.index, b.index, b.phase, move, turn, (unsigned long long)compared, pct[0], pct[1], pct[2], pct[3],
         pct[4], rt_same, m(b, "screen_shadows_rt").c_str(), m(b, "screen_shadows_game").c_str(), m(b, "tlas_instances").c_str(), m(b, "new").c_str(), m(b, "evicted").c_str());
      printf("               sun view flipped %s\n", sun_text);
   }
   // Per frame mesh probes: the same mesh drawn by the camera and by a shadow map must land in the same place
   {
      struct Pair { double max_gap = 0; int captures = 0; };
      std::map<uint32_t, Pair> pairs;
      for (const Capture& c : captures)
      {
         std::map<uint32_t, std::vector<const Capture::Probe*>> by_key;
         for (const auto& p : c.probes)
            if (!(p.flags & 2)) // Not skinned
               by_key[p.key].push_back(&p);
         for (const auto& [key, list] : by_key)
         {
            double gap = -1;
            for (const auto* x : list)
               for (const auto* y : list)
                  if ((x->flags & 1) && !(y->flags & 1))
                  {
                     const double d = std::sqrt((x->pos[0] - y->pos[0]) * (x->pos[0] - y->pos[0]) + (x->pos[1] - y->pos[1]) * (x->pos[1] - y->pos[1]) + (x->pos[2] - y->pos[2]) * (x->pos[2] - y->pos[2]));
                     gap = std::max(gap, d);
                  }
            if (gap >= 0)
            {
               pairs[key].max_gap = std::max(pairs[key].max_gap, gap);
               pairs[key].captures++;
            }
         }
      }
      size_t probes = 0;
      for (const Capture& c : captures)
         probes += c.probes.size();
      printf("\nopenables (dynamic per frame meshes) seen from both the camera and a shadow map in the same capture: %zu meshes (%zu probes)\n", pairs.size(), probes);
      for (const auto& [key, p] : pairs)
         printf("  mesh %08X: in %d captures, largest gap between the shadow map copy and the camera copy %.3f m%s\n", key, p.captures, p.max_gap, p.max_gap > 0.05 ? "  <-- MISPLACED" : "");
   }
   const char* names[5] = { "game_shadow", "rt_raw", "rt_denoised", "scene", "final" };
   for (int phase = 0; phase < 2; phase++)
   {
      const Sum& s = sums[phase];
      if (!s.pairs)
         continue;
      printf("\nphase %d (%s), mean %% of compared pixels flipping between captures, over %d pairs:\n", phase, phase ? "ray tracing scene frozen" : "as is", s.pairs);
      for (int k = 0; k < 5; k++)
         printf("  %-12s %6.2f\n", names[k], s.flips[k] / s.pairs);
      printf("  rt_denoised where the scene class is unchanged: %6.2f\n", s.rt_same_scene / s.pairs);
   }
   printf("\nReading: the game's shadow is the stable reference. scene flips = casters appearing/vanishing in the ray tracing scene;\n"
          "rt flips with the scene unchanged = the trace or the denoiser; final flips with rt stable = the game not using the RT result.\n");
   return 0;
}
