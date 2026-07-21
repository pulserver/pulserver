/**
 * @file recon.hpp
 * @brief RAII C++11 wrapper around pulseg_recon_cache.
 *
 * New-consumer counterpart to mrdserver::read_sequence_cache()
 * (cxx/recon/trajectory_cache_reader.h) -- both are thin wrappers over the
 * same C89 reader, pulseg_recon_cache_read() (csrc/src/recon/pulseg_recon.c,
 * Stage 4 of the vendor-neutral refactor). Use this one for new C++
 * consumers that want the raw pulseg_recon_cache C structs with
 * pulseg::Collection-style RAII, rather than the ISMRMRD-flavoured
 * mrdserver::SequenceCache std:: containers.
 */

#ifndef PULSEG_RECON_HPP
#define PULSEG_RECON_HPP

#include <cstring>
#include <stdexcept>
#include <string>

#include "pulseg_recon.h"

namespace pulseg
{

/**
     * Owning wrapper around a pulseg_recon_cache with RAII lifetime.
     * Movable, not copyable.
     */
class ReconCache
{
  public:
    /**
         * Read the recon-relevant sections (DEFINITIONS/TRAJECTORY/SEQDESC)
         * from a pulseg binary cache file.
         *
         * @throws std::runtime_error if the file cannot be opened, or if
         *         TRAJECTORY or SEQDESC is missing/malformed -- both are
         *         mandatory as of cache format v2.0.0 (D11): there is no
         *         optional-section tolerance to silently degrade past.
         */
    explicit ReconCache(const std::string &cache_path)
    {
        char diag[256];
        const int status = pulseg_recon_cache_read(&cache_, cache_path.c_str(), diag, sizeof(diag));
        if (status != PULSEG_SUCCESS)
            throw std::runtime_error(std::string("pulseg_recon_cache_read failed: ") + diag);
    }

    ~ReconCache()
    {
        pulseg_recon_cache_free(&cache_);
    }

    // Move-only
    ReconCache(ReconCache &&o) noexcept : cache_(o.cache_)
    {
        std::memset(&o.cache_, 0, sizeof(o.cache_));
    }
    ReconCache &operator=(ReconCache &&o) noexcept
    {
        if (this != &o)
        {
            pulseg_recon_cache_free(&cache_);
            cache_ = o.cache_;
            std::memset(&o.cache_, 0, sizeof(o.cache_));
        }
        return *this;
    }
    ReconCache(const ReconCache &) = delete;
    ReconCache &operator=(const ReconCache &) = delete;

    // ── Raw handle (for advanced use) ────────────────────────────
    const pulseg_recon_cache *handle() const
    {
        return &cache_;
    }

    // ── Convenience accessors ────────────────────────────────────
    const pulseg_trajectory &trajectory() const
    {
        return cache_.trajectory;
    }
    const pulseg_sequence_parameters &seq_params() const
    {
        return cache_.seq_params;
    }
    int num_seq_descs() const
    {
        return cache_.num_seq_descs;
    }
    const pulseg_recon_seq_desc &seq_desc(int i) const
    {
        return cache_.seq_descs[i];
    }
    int num_subsequences() const
    {
        return cache_.num_subsequences;
    }

    /** Look up a per-subsequence [DEFINITIONS] entry by key (linear scan). */
    const pulseq_definition *find_definition(int subseq_idx, const char *name) const
    {
        return pulseg_recon_find_definition(&cache_, subseq_idx, name);
    }

  private:
    pulseg_recon_cache cache_{};
};

} // namespace pulseg

#endif // PULSEG_RECON_HPP
