// test_seq_desc_truth.cpp
//
// Parameterized gtest that, for every pulseg cache fixture in
// MRDSERVER_FIXTURES_DIR with a matching <stem>_seq_desc.truth truth
// companion:
//
//   1. opens the cache .pge
//   2. parses its 7-int header + section index
//   3. locates section 7 (SEQDESC) by id
//   4. reads exactly section.size bytes starting at section.offset
//   5. reads the entire truth <stem>_seq_desc.truth payload
//   6. parses both SEQDESC wire-format payloads in parallel and
//      asserts per-field equality:
//        - integers: exact
//        - floats:   |a-b| <= max(SEQDESC_FLOAT_ABS_TOL,
//                                  SEQDESC_FLOAT_REL_TOL * max(|a|,|b|))
//
// Wire format (compact per-block row table):
//   [header]  float x5 + int x4
//   [per-subseq]
//     subseq_idx (int), tr_duration_us (float), num_rows (int)
//     [rows] type (int), timestamp_us (float), params[7] (float x7)
//       RF params[6] = ss_grad_amp_hz_per_m (0 if no slice-select grad)
//
// Tolerant float comparison is used because pulseg stores parsed
// shape/amplitude values in single precision while MATLAB Pulseq
// integrates in double precision before single-cast on write; the two
// pipelines therefore agree to within a few ULPs but not byte-for-byte.
//
// Fixtures lacking a _seq_desc.truth companion are silently skipped.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

    constexpr int32_t CACHE_ENDIAN_MARKER = 0x01020304;
    constexpr int SECTION_SEQUENCEDESCRIPTION = 7;
    constexpr int PULSEG_SEQ_EVENT_PARAMS = 7;

    /* Tolerances for tolerant float comparison.
     *   relative: 1e-5  (~10 ULP of single precision)
     *   absolute: 1e-3  (matches truth float-print precision)
     * These accommodate the ULP-level drift caused by pulseg carrying
     * shape/amplitude values in single precision while MATLAB Pulseq holds
     * them in double precision until the final single-cast on write. */
    constexpr float SEQDESC_FLOAT_REL_TOL = 1e-5f;
    constexpr float SEQDESC_FLOAT_ABS_TOL = 1e-3f;

    void swap4(void *p)
    {
        auto *b = static_cast<uint8_t *>(p);
        std::swap(b[0], b[3]);
        std::swap(b[1], b[2]);
    }

    bool read4_one(std::ifstream &f, int32_t *out, bool do_swap)
    {
        if (!f.read(reinterpret_cast<char *>(out), 4))
            return false;
        if (do_swap)
            swap4(out);
        return true;
    }

    std::vector<fs::path> discover_seqdesc_pairs()
    {
        std::vector<fs::path> out;
        const fs::path dir(MRDSERVER_FIXTURES_DIR);
        if (!fs::is_directory(dir))
            return out;

        static const char *const truth_suffixes[] = {
            "_trajectory",
            "_tr_waveform",
            "_scan_table",
            "_segment_def",
            "_freqmod_def",
            "_freqmod_plan",
            "_label_state",
            "_seq_desc",
        };
        for (const auto &entry : fs::directory_iterator(dir))
        {
            if (!entry.is_regular_file())
                continue;
            const auto &p = entry.path();
            if (p.extension() != ".pge")
                continue;
            const std::string stem = p.stem().string();

            bool is_truth = false;
            for (const char *suf : truth_suffixes)
            {
                const size_t sl = std::string(suf).size();
                if (stem.size() >= sl &&
                    stem.compare(stem.size() - sl, std::string::npos, suf) == 0)
                {
                    is_truth = true;
                    break;
                }
            }
            if (is_truth)
                continue;

            // Require companion _seq_desc.truth on disk.
            const fs::path truth = p.parent_path() / (stem + "_seq_desc.truth");
            if (!fs::exists(truth))
                continue;

            out.push_back(p);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    struct SeqDescSection
    {
        int32_t offset = 0;
        int32_t size = 0;
        bool found = false;
    };

    SeqDescSection locate_seqdesc_section(const fs::path &cache_path)
    {
        SeqDescSection out;

        std::ifstream f(cache_path, std::ios::binary);
        if (!f.is_open())
            return out;

        int32_t marker = 0;
        if (!f.read(reinterpret_cast<char *>(&marker), 4))
            return out;

        bool do_swap = false;
        if (marker != CACHE_ENDIAN_MARKER)
        {
            swap4(&marker);
            if (marker != CACHE_ENDIAN_MARKER)
                return out;
            do_swap = true;
        }

        int32_t version_major, version_minor, version_revision, vendor, stored_size, num_sections;
        if (!read4_one(f, &version_major, do_swap))
            return out;
        if (!read4_one(f, &version_minor, do_swap))
            return out;
        if (!read4_one(f, &version_revision, do_swap))
            return out;
        if (!read4_one(f, &vendor, do_swap))
            return out;
        if (!read4_one(f, &stored_size, do_swap))
            return out;
        if (!read4_one(f, &num_sections, do_swap))
            return out;
        if (num_sections <= 0 || num_sections > 32)
            return out;

        for (int i = 0; i < num_sections; ++i)
        {
            int32_t id, off, sz;
            if (!read4_one(f, &id, do_swap))
                return out;
            if (!read4_one(f, &off, do_swap))
                return out;
            if (!read4_one(f, &sz, do_swap))
                return out;
            if (id == SECTION_SEQUENCEDESCRIPTION)
            {
                out.offset = off;
                out.size = sz;
                out.found = true;
            }
        }
        return out;
    }

    std::vector<uint8_t> read_blob(const fs::path &p, long offset, long size)
    {
        std::vector<uint8_t> buf(static_cast<size_t>(size));
        std::ifstream f(p, std::ios::binary);
        if (!f.is_open())
            return {};
        f.seekg(offset, std::ios::beg);
        if (!f.good())
            return {};
        if (!f.read(reinterpret_cast<char *>(buf.data()), size))
            return {};
        return buf;
    }

    std::vector<uint8_t> read_all(const fs::path &p)
    {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f.is_open())
            return {};
        const auto sz = f.tellg();
        if (sz < 0)
            return {};
        std::vector<uint8_t> buf(static_cast<size_t>(sz));
        f.seekg(0, std::ios::beg);
        if (!f.read(reinterpret_cast<char *>(buf.data()), sz))
            return {};
        return buf;
    }

    class SeqDescTruthFixture : public ::testing::TestWithParam<fs::path>
    {
    };

    /* ------------------------------------------------------------------ */
    /*  Tolerant per-field comparator                                      */
    /* ------------------------------------------------------------------ */

    struct CompareResult
    {
        bool ok = true;
        std::string error;
        std::string field;
        size_t cache_off = 0;
        size_t truth_off = 0;
    };

    class SeqDescStream
    {
    public:
        SeqDescStream(const std::vector<uint8_t> &buf, const char *name)
            : buf_(buf), name_(name) {}

        bool ok() const { return !overflow_; }
        size_t pos() const { return pos_; }
        size_t remaining() const { return overflow_ ? 0 : buf_.size() - pos_; }
        const char *name() const { return name_; }

        int32_t read_i32()
        {
            if (overflow_ || pos_ + 4 > buf_.size())
            {
                overflow_ = true;
                return 0;
            }
            int32_t v;
            std::memcpy(&v, buf_.data() + pos_, 4);
            pos_ += 4;
            return v;
        }
        float read_f32()
        {
            if (overflow_ || pos_ + 4 > buf_.size())
            {
                overflow_ = true;
                return 0.f;
            }
            float v;
            std::memcpy(&v, buf_.data() + pos_, 4);
            pos_ += 4;
            return v;
        }

    private:
        const std::vector<uint8_t> &buf_;
        const char *name_;
        size_t pos_ = 0;
        bool overflow_ = false;
    };

    bool floats_close(float a, float b)
    {
        if (std::isnan(a) && std::isnan(b))
            return true;
        if (a == b)
            return true;
        const float diff = std::fabs(a - b);
        if (diff <= SEQDESC_FLOAT_ABS_TOL)
            return true;
        const float scale = std::max(std::fabs(a), std::fabs(b));
        return diff <= SEQDESC_FLOAT_REL_TOL * scale;
    }

#define COMPARE_INT(field_label)                                          \
    do                                                                    \
    {                                                                     \
        int32_t cv = c.read_i32();                                        \
        int32_t tv = t.read_i32();                                        \
        if (!c.ok() || !t.ok())                                           \
        {                                                                 \
            std::ostringstream os;                                        \
            os << "Truncated stream while reading int '" << (field_label) \
               << "' (cache.pos=" << c.pos() << " truth.pos=" << t.pos()  \
               << ")";                                                    \
            out.ok = false;                                               \
            out.error = os.str();                                         \
            out.field = (field_label);                                    \
            out.cache_off = c.pos();                                      \
            out.truth_off = t.pos();                                      \
            return out;                                                   \
        }                                                                 \
        if (cv != tv)                                                     \
        {                                                                 \
            std::ostringstream os;                                        \
            os << "Int field '" << (field_label) << "' differs: cache="   \
               << cv << " truth=" << tv;                                  \
            out.ok = false;                                               \
            out.error = os.str();                                         \
            out.field = (field_label);                                    \
            out.cache_off = c.pos();                                      \
            out.truth_off = t.pos();                                      \
            return out;                                                   \
        }                                                                 \
    } while (0)

#define READ_INT_BOTH(label, cvar, tvar)                                    \
    do                                                                      \
    {                                                                       \
        cvar = c.read_i32();                                                \
        tvar = t.read_i32();                                                \
        if (!c.ok() || !t.ok())                                             \
        {                                                                   \
            std::ostringstream os;                                          \
            os << "Truncated stream while reading int '" << (label) << "'"; \
            out.ok = false;                                                 \
            out.error = os.str();                                           \
            out.field = (label);                                            \
            out.cache_off = c.pos();                                        \
            out.truth_off = t.pos();                                        \
            return out;                                                     \
        }                                                                   \
        if (cvar != tvar)                                                   \
        {                                                                   \
            std::ostringstream os;                                          \
            os << "Int field '" << (label) << "' differs: cache="           \
               << cvar << " truth=" << tvar;                                \
            out.ok = false;                                                 \
            out.error = os.str();                                           \
            out.field = (label);                                            \
            out.cache_off = c.pos();                                        \
            out.truth_off = t.pos();                                        \
            return out;                                                     \
        }                                                                   \
    } while (0)

#define COMPARE_FLOAT(field_label)                                          \
    do                                                                      \
    {                                                                       \
        float cv = c.read_f32();                                            \
        float tv = t.read_f32();                                            \
        if (!c.ok() || !t.ok())                                             \
        {                                                                   \
            std::ostringstream os;                                          \
            os << "Truncated stream while reading float '" << (field_label) \
               << "'";                                                      \
            out.ok = false;                                                 \
            out.error = os.str();                                           \
            out.field = (field_label);                                      \
            out.cache_off = c.pos();                                        \
            out.truth_off = t.pos();                                        \
            return out;                                                     \
        }                                                                   \
        if (!floats_close(cv, tv))                                          \
        {                                                                   \
            std::ostringstream os;                                          \
            os.precision(9);                                                \
            os << "Float field '" << (field_label)                          \
               << "' differs: cache=" << cv << " truth=" << tv              \
               << " |diff|=" << std::fabs(cv - tv);                         \
            out.ok = false;                                                 \
            out.error = os.str();                                           \
            out.field = (field_label);                                      \
            out.cache_off = c.pos();                                        \
            out.truth_off = t.pos();                                        \
            return out;                                                     \
        }                                                                   \
    } while (0)

    CompareResult compare_seqdesc(const std::vector<uint8_t> &cache_bytes,
                                  const std::vector<uint8_t> &truth_bytes)
    {
        CompareResult out;
        SeqDescStream c(cache_bytes, "cache");
        SeqDescStream t(truth_bytes, "truth");

        /* Header: 5 floats + 4 ints */
        COMPARE_FLOAT("min_te_us");
        COMPARE_FLOAT("min_tr_us");
        COMPARE_FLOAT("max_tr_us");
        COMPARE_FLOAT("max_flip_angle_deg");
        c.read_f32();
        t.read_f32(); /* total_scan_time_us: not used by livesdk, skip */
        int32_t num_subseqs_c, num_subseqs_t;
        READ_INT_BOTH("num_subseqs", num_subseqs_c, num_subseqs_t);
        COMPARE_INT("reserved[0]");
        COMPARE_INT("reserved[1]");
        COMPARE_INT("reserved[2]");

        for (int ss = 0; ss < num_subseqs_c; ++ss)
        {
            std::ostringstream pfx_ss;
            pfx_ss << "subseq[" << ss << "].";
            const std::string sp = pfx_ss.str();

            COMPARE_INT(sp + "subseq_idx");
            COMPARE_FLOAT(sp + "tr_duration_us");

            int32_t num_rows_c, num_rows_t;
            READ_INT_BOTH(sp + "num_rows", num_rows_c, num_rows_t);

            for (int ri = 0; ri < num_rows_c; ++ri)
            {
                std::ostringstream pfx_r;
                pfx_r << sp << "row[" << ri << "].";
                const std::string rp = pfx_r.str();

                COMPARE_INT(rp + "type");
                COMPARE_FLOAT(rp + "timestamp_us");
                for (int p = 0; p < PULSEG_SEQ_EVENT_PARAMS; ++p)
                {
                    std::ostringstream f;
                    f << rp << "params[" << p << "]";
                    COMPARE_FLOAT(f.str());
                }
            }
        }

        if (c.remaining() != t.remaining())
        {
            std::ostringstream os;
            os << "Trailing bytes differ: cache.remaining=" << c.remaining()
               << " truth.remaining=" << t.remaining();
            out.ok = false;
            out.error = os.str();
            out.field = "<trailer>";
            out.cache_off = c.pos();
            out.truth_off = t.pos();
        }
        return out;
    }

    TEST_P(SeqDescTruthFixture, CacheSection5MatchesTruthPerField)
    {
        const fs::path cache_path = GetParam();
        const fs::path truth_path =
            cache_path.parent_path() / (cache_path.stem().string() + "_seq_desc.truth");

        ASSERT_TRUE(fs::exists(truth_path)) << truth_path;

        SeqDescSection sec = locate_seqdesc_section(cache_path);
        ASSERT_TRUE(sec.found)
            << "Section 7 (SEQDESC) not present in " << cache_path;
        ASSERT_GT(sec.size, 0) << "Section 7 (SEQDESC) has zero size in " << cache_path;

        std::vector<uint8_t> cache_bytes = read_blob(cache_path, sec.offset, sec.size);
        std::vector<uint8_t> truth_bytes = read_all(truth_path);

        ASSERT_EQ(cache_bytes.size(), static_cast<size_t>(sec.size));
        ASSERT_FALSE(truth_bytes.empty()) << truth_path;

        CompareResult result = compare_seqdesc(cache_bytes, truth_bytes);
        EXPECT_TRUE(result.ok)
            << "Mismatch in seq_desc payload for " << cache_path.stem()
            << ": " << result.error
            << " (cache_off=" << result.cache_off
            << " truth_off=" << result.truth_off << ")";
    }

    INSTANTIATE_TEST_SUITE_P(
        AllFixtures,
        SeqDescTruthFixture,
        ::testing::ValuesIn(discover_seqdesc_pairs()),
        [](const ::testing::TestParamInfo<fs::path> &info)
        {
            return info.param.stem().string();
        });

} // namespace
