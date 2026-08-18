/**
 * @file file.hpp
 * @brief RAII C++11 wrapper around pulseq_file / pulseq_file_set.
 *
 * pulseg ships its own from-scratch Pulseq `.seq` parser (src/c/pulseq,
 * self-contained, no pulseg dependency), including the pulserver-specific
 * custom LABELSET names (PMC, TRID). File/FileSet expose
 * that native reader as a first-class C++ citizen, for consumers that want
 * the raw Pulseq file model directly -- inspecting blocks/definitions,
 * writing tooling -- without building the full pulseg::Collection
 * intermediate representation.
 */

#ifndef PULSEQ_CXX_FILE_HPP
#define PULSEQ_CXX_FILE_HPP

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

#include "pulseq.h"
#include "pulseq_types.h"

#include "error.hpp"
#include "types.hpp"

namespace pulseq
{

    /**
 * Owning wrapper around a pulseq_file with RAII lifetime.
 * Movable, not copyable.
 */
    class File
    {
    public:
        /** Zero-initialize, applying @p raster as the fallback for whatever
     *  rasters the file itself omits from [DEFINITIONS]. */
        explicit File(const Raster& raster = Raster())
        {
            pulseq_raster r = raster.to_c();
            pulseq_file_init(&file_, &r);
        }

        ~File()
        {
            pulseq_file_free(&file_);
        }

        // Move-only
        File(File&& o) noexcept : file_(o.file_)
        {
            std::memset(&o.file_, 0, sizeof(o.file_));
        }
        File& operator=(File&& o) noexcept
        {
            if (this != &o)
            {
                pulseq_file_free(&file_);
                file_ = o.file_;
                std::memset(&o.file_, 0, sizeof(o.file_));
            }
            return *this;
        }
        File(const File&) = delete;
        File& operator=(const File&) = delete;

        // ── Reading (named constructors) ─────────────────────────────

        /** Parse a .seq file from disk. */
        static File read(const std::string& file_path, const Raster& raster = Raster())
        {
            File f(raster);
            check(pulseq_read(&f.file_, file_path.c_str()));
            return f;
        }

        /** Parse a .seq file already open on a FILE* (e.g. a tmpfile()
     *  populated from an in-memory buffer). */
        static File read_from_buffer(FILE* f_ptr, const Raster& raster = Raster())
        {
            File f(raster);
            check(pulseq_read_from_buffer(&f.file_, f_ptr));
            return f;
        }

        /** Parse only [DEFINITIONS]: enough for TotalDuration, rasters, and the
     *  "next sequence" chain link, without decoding any library. */
        static File read_definitions_only(
            const std::string& file_path,
            const Raster& raster = Raster())
        {
            File f(raster);
            check(pulseq_read_definitions_only(&f.file_, file_path.c_str()));
            return f;
        }

        /** Verify a .seq file's embedded [SIGNATURE] MD5 hash. */
        static bool verify_signature(const std::string& file_path)
        {
            return PULSEQ_SUCCEEDED(pulseq_verify_signature(file_path.c_str()));
        }

        // ── Raw handle (for advanced use) ────────────────────────────

        pulseq_file* handle()
        {
            return &file_;
        }
        const pulseq_file* handle() const
        {
            return &file_;
        }

        // ── Accessors ─────────────────────────────────────────────────

        int num_blocks() const
        {
            return file_.num_blocks;
        }
        int version_combined() const
        {
            return file_.version_combined;
        }
        Raster design_raster() const
        {
            return Raster::from_c(file_.design_raster);
        }
        const pulseq_reserved_definitions& reserved_definitions() const
        {
            return file_.reserved_definitions_library;
        }

        // ── Block accessors ───────────────────────────────────────────

        /** Resolve a block's raw content ids (rf/gx/gy/gz/adc/extension chain
     *  head) from the BLOCKS table, without inlining event data. */
        pulseq_raw_block raw_block_content_ids(int block_index, bool parse_extensions = true) const
        {
            pulseq_raw_block block;
            std::memset(&block, 0, sizeof(block));
            check(pulseq_get_raw_block_content_ids(
                &file_,
                &block,
                block_index,
                parse_extensions ? 1 : 0));
            return block;
        }

        /** Walk a block's extension chain into a resolved index set. */
        pulseq_raw_extension raw_extension(const pulseq_raw_block& raw) const
        {
            pulseq_raw_extension ext;
            std::memset(&ext, 0, sizeof(ext));
            pulseq_get_raw_extension(&file_, &ext, &raw);
            return ext;
        }

    private:
        pulseq_file file_{};
    };

    /**
 * Owning wrapper around a pulseq_file_set (Pulseq "next sequence" chain)
 * with RAII lifetime. Movable, not copyable.
 */
    class FileSet
    {
    public:
        /** Parse a (possibly "next sequence"-chained) .seq file into a set. */
        static FileSet read(const std::string& first_file_path, const Raster& raster = Raster())
        {
            FileSet fs;
            pulseq_raster r = raster.to_c();
            check(pulseq_file_set_read(&fs.set_, first_file_path.c_str(), &r));
            return fs;
        }

        ~FileSet()
        {
            pulseq_file_set_free(&set_);
        }

        // Move-only
        FileSet(FileSet&& o) noexcept : set_(o.set_)
        {
            std::memset(&o.set_, 0, sizeof(o.set_));
        }
        FileSet& operator=(FileSet&& o) noexcept
        {
            if (this != &o)
            {
                pulseq_file_set_free(&set_);
                set_ = o.set_;
                std::memset(&o.set_, 0, sizeof(o.set_));
            }
            return *this;
        }
        FileSet(const FileSet&) = delete;
        FileSet& operator=(const FileSet&) = delete;

        // ── Raw handle (for advanced use) ────────────────────────────

        pulseq_file_set* handle()
        {
            return &set_;
        }
        const pulseq_file_set* handle() const
        {
            return &set_;
        }

        // ── Accessors ─────────────────────────────────────────────────

        int num_sequences() const
        {
            return set_.num_sequences;
        }

        /** The i-th sequence's raw model (0-based, chain order). */
        const pulseq_file& sequence(int i) const
        {
            return set_.sequences[i];
        }

    private:
        FileSet() = default;

        pulseq_file_set set_{};
    };

    // ── Path helpers (for following a "next sequence" chain) ────────────

    /** Directory part of @p file_path, without the trailing separator. */
    inline std::string path_dirname(const std::string& file_path)
    {
        char* raw = pulseq_path_dirname(file_path.c_str());
        if (!raw)
            return std::string();
        std::string out(raw);
        PULSEQ_FREE(raw);
        return out;
    }

    /** Join a base directory and a file name. */
    inline std::string path_join(const std::string& base_path, const std::string& filename)
    {
        char* raw = pulseq_path_join(base_path.c_str(), filename.c_str());
        if (!raw)
            return std::string();
        std::string out(raw);
        PULSEQ_FREE(raw);
        return out;
    }

    // ── Shape codec ───────────────────────────────────────────────────

    /** Decompress a run-length-encoded SHAPES library entry. */
    inline Shape decompress_shape(const pulseq_shape& encoded, float scale = 1.0f)
    {
        pulseq_shape result = PULSEQ_SHAPE_INIT;
        if (!pulseq_decompress_shape(&result, &encoded, scale))
            throw Error(PULSEQ_ERR_INVALID_ARGUMENT);

        Shape out;
        out.num_uncompressed_samples = result.num_uncompressed_samples;
        if (result.samples && result.num_samples > 0)
            out.samples.assign(result.samples, result.samples + result.num_samples);
        PULSEQ_FREE(result.samples);
        return out;
    }

    // ── Name tables ───────────────────────────────────────────────────

    /** Map a label/flag name (e.g. "SLC", "PMC", "TRID") to its numeric id, or
 *  return false if @p name is not a recognized label/flag. */
    inline bool label_id_for_name(const std::string& name, LabelId* out)
    {
        int id = pulseq_label_id_for_name(name.c_str());
        if (id < 0)
            return false;
        if (out)
            *out = static_cast<LabelId>(id);
        return true;
    }

    /** Map a soft-delay hint name (e.g. "TE", "TR", "TI") to its numeric id, or
 *  return false if @p name is not a recognized hint. */
    inline bool hint_id_for_name(const std::string& name, HintId* out)
    {
        int id = pulseq_hint_id_for_name(name.c_str());
        if (id < 0)
            return false;
        if (out)
            *out = static_cast<HintId>(id);
        return true;
    }

} // namespace pulseq

#endif // PULSEQ_CXX_FILE_HPP
