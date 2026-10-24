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
   int combo = 0; // Test matrix combo (bit 0: openables from the shadow map off, bit 1: the old shadow map retention rule)
   bool settle = false; // Right after a combo switch: not compared
   uint32_t w = 0, h = 0;
   std::vector<float> depth;
   std::map<std::string, std::vector<uint8_t>> gray; // game_shadow, rt_raw, rt_denoised
   std::vector<uint8_t> scene, final_image, scene_sun; // RGB
   std::vector<uint16_t> blocker; // TLAS index + 1 of the first hit of a hard sun ray, 0 lit
   struct TlasRow { std::string key; int sources = 0, min_slice = 255; double pos[3] = {}; int indices = 0, vertices = 0; };
   std::vector<TlasRow> tlas; // cNNNN_tlas.csv, row r = blocker id r + 1
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
   if (c.meta.count("combo"))
      c.combo = std::stoi(c.meta["combo"]);
   if (c.meta.count("settle"))
      c.settle = c.meta["settle"] == "1";
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
   std::vector<uint8_t> bl = ReadBytes(dir / (std::string(name) + "_blocker.bin"));
   if (bl.size() == n * 2)
   {
      c.blocker.resize(n);
      std::memcpy(c.blocker.data(), bl.data(), bl.size());
   }
   {
      std::ifstream tl(dir / (std::string(name) + "_tlas.csv"));
      std::string row;
      std::getline(tl, row); // header
      while (std::getline(tl, row))
      {
         Capture::TlasRow t;
         char key[32] = {};
         unsigned first = 0, last = 0, seen = 0;
         if (sscanf(row.c_str(), "%31[^,],%d,%d,%lf,%lf,%lf,%d,%d,%u,%u,%u", key, &t.sources, &t.min_slice, &t.pos[0], &t.pos[1], &t.pos[2], &t.indices, &t.vertices, &first, &last, &seen) >= 8)
         {
            t.key = key;
            c.tlas.push_back(t);
         }
      }
   }
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
      double max_rt = 0;
      int pairs = 0, spikes = 0; // spikes: pairs with more than 5 % of rt_denoised flipping
   };
   std::map<int, Sum> sums; // phase * 10 + combo
   for (size_t k = 1; k < captures.size(); k++)
   {
      const Capture& a = captures[k - 1];
      const Capture& b = captures[k];
      if (a.phase != b.phase || a.combo != b.combo || a.settle || b.settle || a.w != b.w || a.h != b.h)
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
      Sum& sum = sums[(b.phase & 1) * 10 + b.combo];
      for (int s = 0; s < 5; s++)
         sum.flips[s] += pct[s];
      sum.rt_same_scene += rt_same;
      sum.max_rt = std::max(sum.max_rt, pct[2]);
      sum.spikes += pct[2] > 5.0;
      sum.pairs++;
      auto m = [&](const Capture& c, const char* key) { auto it = c.meta.find(key); return it != c.meta.end() ? it->second : std::string("?"); };
      printf("  c%04d-c%04d %d/%d  %6.2f   %6.1f  %7llu | %15.2f %7.2f %7.2f %7.2f %7.2f | %29.2f | %10s/%-4s | %5s %4s %5s\n", a.index, b.index, b.phase, b.combo, move, turn, (unsigned long long)compared, pct[0], pct[1], pct[2], pct[3],
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
   // Against the game's own shadow, per capture (no pairs, so nothing is skipped): the lit fraction of the RT shadow vs
   // the game's. A caster the game doesn't have (or the reverse) shows as a gap, however long it lasts.
   auto lit = [](const std::vector<uint8_t>& v) { size_t n = 0; for (uint8_t x : v) n += x > 127; return v.empty() ? -1.0 : double(n) / v.size(); };
   struct Gap { double sum = 0, max = 0; int n = 0, big = 0; };
   std::map<int, Gap> gaps;
   printf("\nlit fraction, RT vs the game's own shadow, per capture (|gap| > 0.05 marked):\n");
   double prev_rt = -1;
   for (const Capture& c : captures)
   {
      auto g = c.gray.find("game_shadow"), r = c.gray.find("rt_denoised");
      if (g == c.gray.end() || r == c.gray.end())
         continue;
      const double gl = lit(g->second), rl = lit(r->second), gap = rl - gl;
      Gap& acc = gaps[(c.phase & 1) * 10 + c.combo];
      acc.sum += std::fabs(gap);
      acc.max = std::max(acc.max, std::fabs(gap));
      acc.n++;
      acc.big += std::fabs(gap) > 0.05;
      if (std::fabs(gap) > 0.05 || (prev_rt >= 0 && std::fabs(rl - prev_rt) > 0.05))
         printf("  c%04d t=%5.1fs phase %d combo %d%s: game %.3f, rt %.3f, gap %+.3f%s\n", c.index, std::stod(c.meta.count("t_ms") ? c.meta.at("t_ms") : std::string("0")) / 1000.0, c.phase, c.combo, c.settle ? " (settle)" : "", gl, rl, gap,
            prev_rt >= 0 && std::fabs(rl - prev_rt) > 0.05 ? "  <-- RT jump" : "");
      prev_rt = rl;
   }
   // Pass/fail metric (absolute, per capture, switch captures included): where the game's own shadow is lit (the
   // window light), the share the RT shadow also lights ("recall"), and where the game is shadowed, the share RT
   // lights anyway ("false light"). Plus the view depth of the game's lit pixels against cascade 0's far depth.
   struct Agree { double recall_sum = 0, recall_min = 2, false_sum = 0; int n = 0, low = 0; };
   std::map<int, Agree> agree;
   printf("\nagreement with the game's own shadow, per capture (recall < 0.8 or false light > 0.05 marked):\n");
   for (const Capture& c : captures)
   {
      auto g = c.gray.find("game_shadow"), r = c.gray.find("rt_denoised");
      if (g == c.gray.end() || r == c.gray.end())
         continue;
      size_t game_lit = 0, both_lit = 0, game_dark = 0, false_lit = 0;
      std::vector<double> lit_z;
      for (size_t i = 0; i < g->second.size(); i++)
      {
         const bool gl = g->second[i] > 127, rl = r->second[i] > 127;
         if (gl)
         {
            game_lit++;
            both_lit += rl;
            double w[3];
            if (lit_z.size() < 20000 && World(c, uint32_t(i % c.w), uint32_t(i / c.w), w))
               lit_z.push_back((w[0] - c.cam[0]) * c.fwd[0] + (w[1] - c.cam[1]) * c.fwd[1] + (w[2] - c.cam[2]) * c.fwd[2]);
         }
         else
         {
            game_dark++;
            false_lit += rl;
         }
      }
      if (game_lit < 200)
         continue; // No window light in view
      const double recall = double(both_lit) / game_lit, false_light = game_dark ? double(false_lit) / game_dark : 0.0;
      Agree& a = agree[(c.phase & 1) * 10 + c.combo];
      a.recall_sum += recall;
      a.recall_min = std::min(a.recall_min, recall);
      a.false_sum += false_light;
      a.n++;
      a.low += recall < 0.8;
      std::nth_element(lit_z.begin(), lit_z.begin() + lit_z.size() / 2, lit_z.end());
      const double median_z = lit_z.empty() ? -1 : lit_z[lit_z.size() / 2];
      double fars[4] = { -1, -1, -1, -1 };
      if (c.meta.count("slice_fars"))
         ParseList(c.meta.at("slice_fars"), fars, 4);
      if (recall < 0.8 || false_light > 0.05)
         printf("  c%04d t=%5.1fs phase %d combo %d%s: recall %.3f, false light %.3f | lit pixels' median view depth %.2f m, cascade 0 ends at %.2f m\n", c.index,
            std::stod(c.meta.count("t_ms") ? c.meta.at("t_ms") : std::string("0")) / 1000.0, c.phase, c.combo, c.settle ? " (settle)" : "", recall, false_light, median_z, fars[0]);
   }
   // Blockers: where the game's own shadow is lit but the RT shadow is dark, what a hard sun ray hits first
   {
      struct Blocker { int pixels = 0, captures = 0; Capture::TlasRow row; int last_capture = -1; };
      std::map<std::string, Blocker> blockers; // By mesh key + rounded position
      int wrong_pixels = 0, unexplained = 0, facing_away = 0;
      for (const Capture& c : captures)
      {
         auto g = c.gray.find("game_shadow"), r = c.gray.find("rt_denoised");
         if (g == c.gray.end() || r == c.gray.end() || c.blocker.empty() || c.tlas.empty())
            continue;
         for (size_t i = 0; i < c.blocker.size(); i++)
         {
            if (!(g->second[i] > 127 && r->second[i] <= 127))
               continue;
            wrong_pixels++;
            const uint16_t id = c.blocker[i];
            if (id == 65534)
            {
               facing_away++;
               continue;
            }
            if (id == 0 || id > c.tlas.size())
            {
               unexplained++;
               continue;
            }
            const Capture::TlasRow& t = c.tlas[id - 1];
            char k[96];
            snprintf(k, sizeof(k), "%s@%.1f,%.1f,%.1f", t.key.c_str(), t.pos[0], t.pos[1], t.pos[2]);
            Blocker& b = blockers[k];
            b.pixels++;
            b.row = t;
            if (b.last_capture != c.index)
            {
               b.captures++;
               b.last_capture = c.index;
            }
         }
      }
      std::vector<std::pair<std::string, Blocker>> sorted(blockers.begin(), blockers.end());
      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.pixels > b.second.pixels; });
      printf("\nblockers of the game's lit pixels that RT darkens: %d pixels, %d with no hard ray hit (penumbra/denoiser), %d facing away from the sun (depth normal)\n", wrong_pixels, unexplained, facing_away);
      for (size_t k = 0; k < sorted.size() && k < 15; k++)
      {
         const auto& b = sorted[k].second;
         const int s = b.row.sources;
         std::string src = std::string((s & 4) ? "per-frame mesh" : "") + ((s & 1) ? " camera" : "") + ((s & 2) ? " shadow-map" : "") + ((s & 8) ? " (MVP)" : "") + ((s & 16) ? " skinned" : "");
         printf("  %6d px in %3d captures: mesh %s, %s, lowest slice %s, %d indices / %d vertices, at %.2f %.2f %.2f\n", b.pixels, b.captures, b.row.key.c_str(), src.c_str(),
            b.row.min_slice == 255 ? "-" : std::to_string(b.row.min_slice).c_str(), b.row.indices, b.row.vertices, b.row.pos[0], b.row.pos[1], b.row.pos[2]);
      }
   }
   const char* combo_names[4] = { "default (near slices only, static shadow world kept)", "near slices only, OLD retention", "ALL slices, OLD retention", "(unused)" };
   printf("\nper combo (pairs within a block, settling captures skipped): mean %% of compared pixels flipping\n");
   printf("  %-58s %5s | %8s %7s %7s %7s %7s | %6s %6s\n", "combo", "pairs", "game_sh", "rt_raw", "rt_den", "scene", "final", "max rt", "spikes");
   printf("\nper combo, agreement with the game's own shadow (captures with window light in view):\n");
   for (const auto& [id, a] : agree)
   {
      const int phase = id / 10, combo = id % 10;
      std::string name = phase ? std::string("FROZEN scene, ") + combo_names[combo & 3] : combo_names[combo & 3];
      printf("  %-58s %3d captures | recall mean %.3f, min %.3f, captures below 0.8: %d | false light mean %.3f\n", name.c_str(), a.n, a.recall_sum / a.n, a.recall_min, a.low, a.false_sum / a.n);
   }
   printf("\nper combo, RT lit fraction vs the game's own shadow (all captures):\n");
   for (const auto& [id, gap] : gaps)
   {
      const int phase = id / 10, combo = id % 10;
      std::string name = phase ? std::string("FROZEN scene, ") + combo_names[combo & 3] : combo_names[combo & 3];
      printf("  %-58s %3d captures | mean |gap| %.3f, max %.3f, captures with |gap| > 0.05: %d\n", name.c_str(), gap.n, gap.sum / gap.n, gap.max, gap.big);
   }
   printf("\n");
   for (const auto& [id, s] : sums)
   {
      if (!s.pairs)
         continue;
      const int phase = id / 10, combo = id % 10;
      std::string name = phase ? std::string("FROZEN scene, ") + combo_names[combo & 3] : combo_names[combo & 3];
      printf("  %-58s %5d | %8.2f %7.2f %7.2f %7.2f %7.2f | %6.2f %6d\n", name.c_str(), s.pairs, s.flips[0] / s.pairs, s.flips[1] / s.pairs, s.flips[2] / s.pairs, s.flips[3] / s.pairs, s.flips[4] / s.pairs, s.max_rt, s.spikes);
   }
   printf("\nReading: the game's shadow is the stable reference. scene flips = casters appearing/vanishing in the ray tracing scene;\n"
          "rt flips with the scene unchanged = the trace or the denoiser; final flips with rt stable = the game not using the RT result.\n");
   return 0;
}
