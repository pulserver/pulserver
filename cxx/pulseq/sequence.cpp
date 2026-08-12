/**
 * @file sequence.cpp
 * @brief Storage and block operations for pulseq::Sequence.
 *
 * See cxx/include/pulseq/sequence.hpp for the model.  Everything here is
 * bookkeeping: appending rows, handing out ids, and reading a block back.
 * The writers, the reader and deduplication live beside this file.
 */

#include "pulseq/sequence.hpp"

#include <numeric>

namespace pulseq
{

    namespace
    {
        const std::string kNoName;

        void require_block(int index, int count)
        {
            if (index < 1 || index > count)
            {
                throw std::out_of_range(
                    "block index " + std::to_string(index) + " is outside 1.." +
                    std::to_string(count));
            }
        }
    }  // namespace

    const std::vector<std::string>& Sequence::builtin_labels()
    {
        // Pulseq's own list, in the order that numbers it.  Deliberately the
        // 22 names PyPulseq exposes and not the 23 the C reader's table
        // carries: the extra one there (OFF) is this project's, and a sequence
        // using it is using a name Pulseq does not define -- which is
        // precisely what raises the file to revision 1.5.2.  TRID is in this
        // list and so costs nothing, which is one reason Pulserver's
        // safety-group label is TRID and not a name of its own.
        static const std::vector<std::string> names{
            "SLC", "SEG", "REP", "AVG", "SET",   "ECO", "PHS",   "LIN",
            "PAR", "ACQ", "NAV", "REV", "SMS",   "REF", "IMA",   "NOISE",
            "PMC", "NOROT", "NOPOS", "NOSCL", "ONCE", "TRID"};
        return names;
    }

    Sequence::Sequence()
    {
        label_names_ = builtin_labels();
        for (size_t i = 0; i < label_names_.size(); ++i)
            label_ids_.emplace(label_names_[i], static_cast<int>(i) + 1);
    }

    int Sequence::find_label_id(const std::string& name) const
    {
        auto it = label_ids_.find(name);
        return it == label_ids_.end() ? 0 : it->second;
    }

    int Sequence::label_id(const std::string& name)
    {
        auto it = label_ids_.find(name);
        if (it != label_ids_.end())
            return it->second;

        label_names_.push_back(name);
        const int id = static_cast<int>(label_names_.size());
        label_ids_.emplace(name, id);
        return id;
    }

    const std::string& Sequence::label_name(int id) const
    {
        if (id < 1 || id > static_cast<int>(label_names_.size()))
            return kNoName;
        return label_names_[id - 1];
    }

    bool Sequence::is_custom_label(int id) const
    {
        return id > static_cast<int>(builtin_labels().size());
    }

    /* ================================================================== */
    /*  Version and rasters                                               */
    /* ================================================================== */

    void Sequence::set_version(int major, int minor, int revision)
    {
        version_major_ = major;
        version_minor_ = minor;
        version_revision_ = revision;
    }

    void Sequence::set_rasters(double rf, double grad, double adc, double block)
    {
        rf_raster_ = rf;
        grad_raster_ = grad;
        adc_raster_ = adc;
        block_raster_ = block;
    }

    /* ================================================================== */
    /*  Definitions                                                       */
    /* ================================================================== */

    void Sequence::set_definition(const std::string& key, Definition value)
    {
        definitions_[key] = std::move(value);
    }

    const Definition* Sequence::definition(const std::string& key) const
    {
        auto it = definitions_.find(key);
        return it == definitions_.end() ? nullptr : &it->second;
    }

    /* ================================================================== */
    /*  Extension type registry                                           */
    /* ================================================================== */

    int Sequence::find_extension_type_id(const std::string& name) const
    {
        auto it = extension_ids_.find(name);
        return it == extension_ids_.end() ? 0 : it->second;
    }

    int Sequence::extension_type_id(const std::string& name)
    {
        auto it = extension_ids_.find(name);
        if (it != extension_ids_.end())
            return it->second;

        // Ids are handed out in first-use order, which is what PyPulseq does
        // and therefore what keeps a written file identical to one it wrote.
        const int id = static_cast<int>(extension_ids_.size()) + 1;
        extension_ids_.emplace(name, id);
        extension_names_.emplace(id, name);
        return id;
    }

    const std::string& Sequence::extension_type_name(int id) const
    {
        auto it = extension_names_.find(id);
        return it == extension_names_.end() ? kNoName : it->second;
    }

    void Sequence::set_extension_type_id(const std::string& name, int id)
    {
        // Reading a file is the case that has to force the mapping: the file
        // says what its own numbering is, and it need not match the order the
        // sections happen to appear in.
        auto existing = extension_ids_.find(name);
        if (existing != extension_ids_.end())
            extension_names_.erase(existing->second);
        extension_ids_[name] = id;
        extension_names_[id] = name;
    }

    /* ================================================================== */
    /*  Event registration                                                */
    /* ================================================================== */

    int Sequence::register_rf(const double* row, char use)
    {
        rf_use_.push_back(use);
        return rf_.append(row);
    }

    int Sequence::register_trap(const double* row)
    {
        const int slot = trap_.append(row);
        grad_slot_.push_back(static_cast<int32_t>(slot));
        return static_cast<int>(grad_slot_.size());
    }

    int Sequence::register_arbitrary(const double* row)
    {
        const int slot = arb_.append(row);
        grad_slot_.push_back(-static_cast<int32_t>(slot));
        return static_cast<int>(grad_slot_.size());
    }

    int Sequence::register_adc(const double* row) { return adc_.append(row); }
    int Sequence::register_trigger(const double* row) { return trigger_.append(row); }
    int Sequence::register_rotation(const double* row) { return rotation_.append(row); }

    int Sequence::register_label_set(int32_t value, int32_t label_id)
    {
        const int32_t row[LABEL_WIDTH] = {value, label_id};
        return label_set_.append(row);
    }

    int Sequence::register_label_inc(int32_t value, int32_t label_id)
    {
        const int32_t row[LABEL_WIDTH] = {value, label_id};
        return label_inc_.append(row);
    }

    int Sequence::register_rf_shim(const double* values, int count)
    {
        return rf_shim_.append(values, count);
    }

    int Sequence::register_soft_delay(const SoftDelay& row)
    {
        soft_delays_.push_back(row);
        return static_cast<int>(soft_delays_.size());
    }

    int Sequence::register_shape(int num_uncompressed, const double* samples, int count)
    {
        return shapes_.append(num_uncompressed, samples, count);
    }

    int Sequence::register_raw_shape(const double* samples, int count)
    {
        return shapes_.append_raw(samples, count);
    }

    void Sequence::compress_shapes() { shapes_.compress(); }

    int Sequence::chain_extension(int32_t type_id, int32_t ref, int32_t next)
    {
        const std::array<int32_t, EXTENSION_WIDTH> key{type_id, ref, next};
        auto it = chain_index_.find(key);
        if (it != chain_index_.end())
            return it->second;

        const int id = extensions_.append(key.data());
        chain_index_.emplace(key, id);
        return id;
    }

    int Sequence::append_extension(int32_t type_id, int32_t ref, int32_t next)
    {
        const std::array<int32_t, EXTENSION_WIDTH> row{type_id, ref, next};
        return extensions_.append(row.data());
    }

    /* ================================================================== */
    /*  Gradients                                                         */
    /* ================================================================== */

    GradKind Sequence::grad_kind(int id) const
    {
        if (id < 1 || id > static_cast<int>(grad_slot_.size()))
            return GradKind::None;
        return grad_slot_[id - 1] > 0 ? GradKind::Trap : GradKind::Arbitrary;
    }

    int Sequence::grad_row(int id) const
    {
        if (id < 1 || id > static_cast<int>(grad_slot_.size()))
            return 0;
        const int32_t slot = grad_slot_[id - 1];
        return slot > 0 ? slot : -slot;
    }

    /* ================================================================== */
    /*  Blocks                                                            */
    /* ================================================================== */

    int Sequence::add_block(const Block& block)
    {
        blocks_.insert(blocks_.end(), {block.rf, block.gx, block.gy, block.gz, block.adc, block.ext});
        durations_.push_back(block.duration);
        return static_cast<int>(durations_.size());
    }

    void Sequence::set_block(int index, const Block& block)
    {
        require_block(index, num_blocks());
        int32_t* row = blocks_.data() + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
        row[0] = block.rf;
        row[1] = block.gx;
        row[2] = block.gy;
        row[3] = block.gz;
        row[4] = block.adc;
        row[5] = block.ext;
        durations_[index - 1] = block.duration;
    }

    Block Sequence::get_block(int index) const
    {
        require_block(index, num_blocks());
        const int32_t* row = blocks_.data() + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
        Block block;
        block.rf = row[0];
        block.gx = row[1];
        block.gy = row[2];
        block.gz = row[3];
        block.adc = row[4];
        block.ext = row[5];
        block.duration = durations_[index - 1];
        return block;
    }

    void Sequence::set_blocks(const int32_t* events, const double* durations, int count)
    {
        blocks_.assign(events, events + static_cast<size_t>(count) * BLOCK_WIDTH);
        durations_.assign(durations, durations + count);
    }

    void Sequence::set_grad_slots(const int32_t* slots, int count)
    {
        grad_slot_.assign(slots, slots + count);
    }

    void Sequence::set_shapes(const int32_t* num_uncompressed, int count, const int32_t* starts,
                              const double* samples)
    {
        shapes_.assign(num_uncompressed, count, starts, samples);
    }

    void Sequence::set_rf_shims(const int32_t* starts, int count, const double* values)
    {
        rf_shim_.assign(starts, count, values);
    }

    namespace
    {
        /**
         * Pairwise summation, as NumPy's `.sum()` does it.
         *
         * Adding a couple of million block durations left to right lets the
         * rounding error grow with the number of terms; summing in a balanced
         * tree makes it grow with the logarithm instead, which on a scan of two
         * million blocks is the difference between an error in the last few
         * bits and one that is visible.
         *
         * The blocking is NumPy's exactly -- eight accumulators up to 128
         * elements, splitting on a multiple of eight above that -- and that is
         * deliberate rather than incidental: `TotalDuration` goes into the
         * binary file as a full float64, so a sequence written here and the
         * same one written through NumPy have to agree to the bit, not just to
         * nine significant figures.
         */
        double pairwise_sum(const double* values, size_t n)
        {
            constexpr size_t BLOCKSIZE = 128;

            if (n < 8)
            {
                double result = 0.0;
                for (size_t i = 0; i < n; ++i)
                    result += values[i];
                return result;
            }

            if (n <= BLOCKSIZE)
            {
                double r[8];
                for (int k = 0; k < 8; ++k)
                    r[k] = values[k];

                size_t i = 8;
                for (; i < n - (n % 8); i += 8)
                {
                    for (int k = 0; k < 8; ++k)
                        r[k] += values[i + k];
                }

                double result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for (; i < n; ++i)
                    result += values[i];
                return result;
            }

            // Halve, but keep the split on a multiple of the unroll factor so
            // both halves take the branch above in the same shape.
            size_t half = n / 2;
            half -= half % 8;
            return pairwise_sum(values, half) + pairwise_sum(values + half, n - half);
        }
    }  // namespace

    double Sequence::duration() const
    {
        return pairwise_sum(durations_.data(), durations_.size());
    }

}  // namespace pulseq
