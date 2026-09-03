/**
 * @file sequence.hpp
 * @brief A Pulseq sequence: event libraries, a block table, and the operations
 *        that build and read them.
 *
 * This is the model PyPulseq keeps as a `Sequence`, held the way a large scan
 * needs it held.  The difference is not the contents -- every library here is
 * the same library the file format defines -- but the shape of the container:
 * each one is a dense array of fixed-width rows rather than a dictionary of
 * tuples, because a three-dimensional protocol has millions of blocks and an
 * object per row is the whole cost of building one.
 *
 * Nothing here knows about Python, and nothing here knows about pulseg.  The
 * reader below it (src/c/pulseq) and this are the two halves of a Pulseq
 * implementation that can be lifted out on their own.
 *
 * ### Gradients
 *
 * A gradient is either a trapezoid (five numbers) or an arbitrary waveform (a
 * shape reference and five more), and the file format numbers both in one
 * sequence -- trapezoid 3 and arbitrary 4 can sit in the same scan.  Keeping
 * them in one ragged table would cost a branch and a variable stride on every
 * read, so they are stored apart, in two tables that are each fixed-width, and
 * `grad_slot_` maps the shared id onto whichever row is real: a positive value
 * indexes the trapezoid table, a negative one the arbitrary table.
 *
 * The shared id is what a block stores and what the file carries, so a
 * sequence written out is numbered exactly as PyPulseq would have numbered it.
 * The split is an implementation detail that never reaches the file.
 *
 * ### Ids
 *
 * Every library is 1-based, and 0 means "no event" wherever a block refers to
 * one.  That is the file format's convention, not a choice made here.
 */

#ifndef PULSEQ_CXX_SEQUENCE_HPP
#define PULSEQ_CXX_SEQUENCE_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulseq
{

    /* ================================================================== */
    /*  Column counts                                                     */
    /* ================================================================== */

    /** amplitude, mag_shape, phase_shape, time_shape, center, delay,
     *  freq_ppm, phase_ppm, freq, phase */
    constexpr int RF_WIDTH = 10;
    /** amplitude, rise, flat, fall, delay */
    constexpr int TRAP_WIDTH = 5;
    /** amplitude, first, last, amp_shape, time_shape, delay */
    constexpr int ARB_WIDTH = 6;
    /** num, dwell, delay, freq_ppm, phase_ppm, freq, phase, phase_shape */
    constexpr int ADC_WIDTH = 8;
    /** type, channel, delay, duration */
    constexpr int TRIGGER_WIDTH = 4;
    /** quaternion, scalar first, canonical */
    constexpr int ROTATION_WIDTH = 4;
    /** type, ref, next */
    constexpr int EXTENSION_WIDTH = 3;
    /** value, label id */
    constexpr int LABEL_WIDTH = 2;
    /** rf, gx, gy, gz, adc, ext -- the block table's event columns */
    constexpr int BLOCK_WIDTH = 6;

    /* ================================================================== */
    /*  Tables                                                            */
    /* ================================================================== */

    /**
     * A library of fixed-width rows, stored as one flat array.
     *
     * Ids are 1-based and handed out in append order, which is what makes a
     * written file reproducible: PyPulseq numbers by first appearance too, so
     * the same sequence built the same way comes out with the same ids.
     */
    template <typename T> class BasicTable
    {
    public:
        explicit BasicTable(int width) : width_(width)
        {
        }

        int width() const
        {
            return width_;
        }
        int size() const
        {
            return width_ ? static_cast<int>(values_.size()) / width_ : 0;
        }
        bool empty() const
        {
            return values_.empty();
        }

        /** Row @p id, 1-based.  Unchecked; callers hold ids they were given. */
        const T* row(int id) const
        {
            return values_.data() + static_cast<size_t>(id - 1) * width_;
        }
        T* row(int id)
        {
            return values_.data() + static_cast<size_t>(id - 1) * width_;
        }

        const T* data() const
        {
            return values_.data();
        }
        T* data()
        {
            return values_.data();
        }

        /** Append one row.  @return its 1-based id. */
        int append(const T* values)
        {
            values_.insert(values_.end(), values, values + width_);
            return size();
        }

        void reserve(int rows)
        {
            values_.reserve(static_cast<size_t>(rows) * width_);
        }

        /** Grow or shrink to @p rows, zero-filling anything new.  Bulk loading
         *  (a composed scan handed over whole) resizes once and writes in. */
        void resize(int rows)
        {
            values_.resize(static_cast<size_t>(rows) * width_, T{});
        }

        void clear()
        {
            values_.clear();
        }

    private:
        std::vector<T> values_;
        int width_;
    };

    using Table = BasicTable<double>;
    using IntTable = BasicTable<int32_t>;

    /**
     * A library whose rows differ in length: shapes, and pTx shim vectors.
     *
     * Kept as one sample array plus a row-start index, so a row is a pointer
     * and a length rather than a vector of its own.
     */
    class RaggedTable
    {
    public:
        RaggedTable() = default;
        RaggedTable(const RaggedTable& other)
        {
            copy_from(other);
        }
        RaggedTable& operator=(const RaggedTable& other)
        {
            if (this != &other)
                copy_from(other);
            return *this;
        }
        RaggedTable(RaggedTable&&) noexcept = default;
        RaggedTable& operator=(RaggedTable&&) noexcept = default;

        int size() const
        {
            return static_cast<int>(len_.size());
        }
        bool empty() const
        {
            return size() == 0;
        }

        int length(int id) const
        {
            return len_[static_cast<size_t>(id) - 1];
        }
        const double* row(int id) const
        {
            return at(start_[static_cast<size_t>(id) - 1]);
        }
        double* row(int id)
        {
            return at(start_[static_cast<size_t>(id) - 1]);
        }

        /** Make room for @p rows rows; the values need no reservation, a
         *  chunk is allocated once and never moved. */
        void reserve(int rows, size_t /*values*/)
        {
            start_.reserve(static_cast<size_t>(rows));
            len_.reserve(static_cast<size_t>(rows));
        }

        /** Append a row of @p count values each divided by @p divisor (1 when
         *  it is 0).  @return its 1-based id. */
        int append_divided(const double* values, int count, double divisor)
        {
            const int64_t where = place(count);
            double* dst = at(where);
            if (divisor != 0.0)
                for (int i = 0; i < count; ++i)
                    dst[i] = values[i] / divisor;
            else if (count > 0)
                std::memcpy(dst, values, static_cast<size_t>(count) * sizeof(double));
            start_.push_back(where);
            len_.push_back(count);
            return size();
        }
        /** Append a row of @p count values.  @return its 1-based id. */
        int append(const double* values, int count)
        {
            const int64_t where = place(count);
            if (count > 0)
                std::memcpy(at(where), values, static_cast<size_t>(count) * sizeof(double));
            start_.push_back(where);
            len_.push_back(count);
            return size();
        }

        void clear()
        {
            chunks_.clear();
            bases_.clear();
            caps_.clear();
            start_.clear();
            len_.clear();
            cursor_ = 0;
        }

        /**
         * Keep the rows flagged in @p keep (indexed by id - 1), in order;
         * @p new_id (indexed by id) receives every row's new id, 0 for one
         * dropped. Rows are reached through their offsets, so nothing moves:
         * a dropped row's bytes stay in their chunk until the table is
         * cleared, and the survivors keep their places.
         */
        void compact(const uint8_t* keep, int32_t* new_id)
        {
            const int total = size();
            int kept = 0;
            for (int id = 1; id <= total; ++id)
            {
                if (!keep[id - 1])
                {
                    new_id[id] = 0;
                    continue;
                }
                start_[static_cast<size_t>(kept)] = start_[static_cast<size_t>(id) - 1];
                len_[static_cast<size_t>(kept)] = len_[static_cast<size_t>(id) - 1];
                new_id[id] = ++kept;
            }
            start_.resize(static_cast<size_t>(kept));
            len_.resize(static_cast<size_t>(kept));
        }

        /**
         * Replace every row at once.
         *
         * @p starts holds @p count + 1 offsets into @p values, so row i spans
         * `[starts[i], starts[i+1])`.
         */
        void assign(const int32_t* starts, int count, const double* values)
        {
            clear();
            reserve(count, 0);
            for (int i = 0; i < count; ++i)
                append(values + starts[i], starts[i + 1] - starts[i]);
        }

    private:
        /** Rows live in chunks allocated once: a chunk fills until the next
         *  row would not fit, and a row longer than a chunk gets one of its
         *  own. Offsets are into one virtual span the chunks tile. */
        static constexpr int64_t kChunkValues = int64_t{1} << 22;

        double* at(int64_t offset)
        {
            const size_t i = chunk_of(offset);
            return chunks_[i].get() + (offset - bases_[i]);
        }
        const double* at(int64_t offset) const
        {
            const size_t i = chunk_of(offset);
            return chunks_[i].get() + (offset - bases_[i]);
        }
        size_t chunk_of(int64_t offset) const
        {
            const auto it = std::upper_bound(bases_.begin(), bases_.end(), offset);
            return static_cast<size_t>(it - bases_.begin()) - 1;
        }
        int64_t place(int count)
        {
            if (!chunks_.empty() && cursor_ - bases_.back() + count <= caps_.back())
            {
                const int64_t where = cursor_;
                cursor_ += count;
                return where;
            }
            const int64_t cap = count > kChunkValues ? count : kChunkValues;
            chunks_.push_back(make_chunk(cap));
            bases_.push_back(cursor_);
            caps_.push_back(cap);
            const int64_t where = cursor_;
            cursor_ += count;
            return where;
        }
        void copy_from(const RaggedTable& other)
        {
            clear();
            reserve(other.size(), 0);
            for (int id = 1; id <= other.size(); ++id)
                append(other.row(id), other.length(id));
        }

        /** A chunk's storage: an anonymous mapping advised onto huge pages
         *  where the platform has them (a 32 MB chunk is then a handful of
         *  page faults instead of eight thousand), plain heap otherwise. */
        struct ChunkDeleter
        {
            size_t bytes = 0;
            bool mapped = false;
            void operator()(double* p) const;
        };
        using Chunk = std::unique_ptr<double[], ChunkDeleter>;
        static Chunk make_chunk(int64_t cap);

        std::vector<Chunk> chunks_;
        std::vector<int64_t> bases_;
        std::vector<int64_t> caps_;
        std::vector<int64_t> start_;
        std::vector<int32_t> len_;
        int64_t cursor_ = 0;
    };

    class ShapeLibrary
    {
    public:
        int size() const
        {
            return data_.size();
        }
        bool empty() const
        {
            return size() == 0;
        }

        int num_uncompressed(int id) const
        {
            return num_uncompressed_[id - 1];
        }
        int num_compressed(int id) const
        {
            return data_.length(id);
        }
        const double* samples(int id) const
        {
            return data_.row(id);
        }
        /** False while a shape is still held as the waveform it was given as. */
        bool is_compressed(int id) const
        {
            return is_compressed_[id - 1] != 0;
        }

        /** Append a shape already in its compressed form.  @return its id. */
        int append(int num_uncompressed, const double* samples, int count);

        /**
         * Append a shape as it stands, to be compressed later.
         *
         * The registration path a scan takes when it is going to deduplicate
         * anyway: compressing is a pass over every sample, and a sequence that
         * registers a waveform per shot registers far more of them than it
         * keeps.  See `compress`.
         */
        int append_raw(const double* samples, int count);
        /** As append_raw for samples that are @p divisor times the row to
         *  store, @p divisor their signed peak: the row's peak is 1 without
         *  a scan. A divisor of 1 stores the samples as they are, scanned. */
        int append_raw_divided(const double* samples, int count, double divisor);

        /** Encode every shape still held raw.  @return whether any row
         *  changed: a shape the encoding would not shorten is kept as it is,
         *  so a library of such shapes is untouched.  Idempotent. */
        bool compress();

        /**
         * Keep only the shapes @p first maps onto themselves (first[id] ==
         * id), in id order, renumbering densely.  @return for every old id
         * the id its first appearance now has.
         */
        std::vector<int32_t> keep_first_appearances(const std::vector<int32_t>& first);

        /**
         * The shape's first sample, last sample and peak magnitude, as
         * decompressed. Recorded when a raw shape is appended; a shape that
         * arrived encoded is decoded once, the first time it is asked.
         */
        void edge_stats(int id, double* first, double* last, double* peak) const;

        void clear()
        {
            num_uncompressed_.clear();
            is_compressed_.clear();
            first_.clear();
            last_.clear();
            peak_.clear();
            data_.clear();
        }

        /** Replace every shape at once, all compressed.  See RaggedTable::assign. */
        void assign(
            const int32_t* num_uncompressed,
            int count,
            const int32_t* starts,
            const double* samples)
        {
            num_uncompressed_.assign(num_uncompressed, num_uncompressed + count);
            first_.assign(static_cast<size_t>(count), std::numeric_limits<double>::quiet_NaN());
            last_.assign(static_cast<size_t>(count), std::numeric_limits<double>::quiet_NaN());
            peak_.assign(static_cast<size_t>(count), std::numeric_limits<double>::quiet_NaN());
            is_compressed_.assign(static_cast<size_t>(count), 1);
            data_.assign(starts, count, samples);
        }

    private:
        std::vector<int32_t> num_uncompressed_;
        std::vector<uint8_t> is_compressed_;
        /** Per shape, as decompressed; NaN until known. Filled at append_raw,
         *  decoded on demand for shapes appended encoded or assigned. */
        mutable std::vector<double> first_;
        mutable std::vector<double> last_;
        mutable std::vector<double> peak_;
        RaggedTable data_;
    };

    /** One soft-delay row: a numeric id, an offset, a factor, and a hint name. */
    struct SoftDelay
    {
        int32_t num = 0;
        double offset = 0.0;
        double factor = 0.0;
        std::string hint;
    };

    /* ================================================================== */
    /*  Definitions                                                       */
    /* ================================================================== */

    /**
     * One `[DEFINITIONS]` entry.
     *
     * Whether a value is text, whole numbers or reals is not decoration: the
     * text writer formats each differently and the binary format tags them,
     * so a definition read as an integer has to be written back as one.
     */
    class Definition
    {
    public:
        enum class Kind
        {
            Text,
            Int,
            Real
        };

        Definition() = default;
        explicit Definition(std::string text) : kind_(Kind::Text), text_(std::move(text))
        {
        }
        explicit Definition(double value) : kind_(Kind::Real), numbers_{value}
        {
        }
        explicit Definition(std::vector<double> values)
            : kind_(Kind::Real), numbers_(std::move(values))
        {
        }
        static Definition integers(std::vector<double> values)
        {
            Definition d;
            d.kind_ = Kind::Int;
            d.numbers_ = std::move(values);
            return d;
        }

        Kind kind() const
        {
            return kind_;
        }
        const std::string& text() const
        {
            return text_;
        }
        const std::vector<double>& numbers() const
        {
            return numbers_;
        }

    private:
        Kind kind_ = Kind::Real;
        std::string text_;
        std::vector<double> numbers_;
    };

    /* ================================================================== */
    /*  Blocks                                                            */
    /* ================================================================== */

    /** One block's event ids and its duration in seconds.  0 means absent. */
    struct Block
    {
        int32_t rf = 0;
        int32_t gx = 0;
        int32_t gy = 0;
        int32_t gz = 0;
        int32_t adc = 0;
        int32_t ext = 0;
        double duration = 0.0;
    };

    /** What a gradient id resolves to. */
    enum class GradKind
    {
        None,
        Trap,
        Arbitrary
    };

    /* ================================================================== */
    /*  The sequence                                                      */
    /* ================================================================== */

    class Sequence
    {
    public:
        Sequence();

        /**
         * Copyable by value, and deliberately so.
         *
         * Every member below is a plain container, so the compiler's copy is
         * both correct and about as cheap as a copy of this much data can
         * be -- one allocation and one memcpy per library.  That is what lets
         * a transform offer "give me the result, leave the original alone"
         * without a bespoke clone that would have to be kept in step with
         * every field added here.  Anything added to this class that owns a
         * resource by raw pointer would silently break that guarantee, which
         * is why these are spelled out rather than left implicit.
         */
        Sequence(const Sequence&) = default;
        Sequence& operator=(const Sequence&) = default;
        Sequence(Sequence&&) noexcept = default;
        Sequence& operator=(Sequence&&) noexcept = default;

        /* -- version ---------------------------------------------------- */

        int version_major() const
        {
            return version_major_;
        }
        int version_minor() const
        {
            return version_minor_;
        }
        int version_revision() const
        {
            return version_revision_;
        }
        void set_version(int major, int minor, int revision);

        /* -- rasters ---------------------------------------------------- */

        double rf_raster_time() const
        {
            return rf_raster_;
        }
        double grad_raster_time() const
        {
            return grad_raster_;
        }
        double adc_raster_time() const
        {
            return adc_raster_;
        }
        double block_duration_raster() const
        {
            return block_raster_;
        }
        void set_rasters(double rf, double grad, double adc, double block);

        /**
         * Record the four rasters in `[DEFINITIONS]`, unless already recorded.
         *
         * Block durations are serialised as raster ticks, so a reader that
         * cannot see the raster cannot recover the seconds: it falls back to
         * a default, and every time in the file comes out scaled. The writers
         * call this.
         */
        void publish_rasters();

        /* -- definitions ------------------------------------------------ */

        /** Sorted by key, which is the order the writers emit them in. */
        const std::map<std::string, Definition>& definitions() const
        {
            return definitions_;
        }
        void set_definition(const std::string& key, Definition value);
        const Definition* definition(const std::string& key) const;

        /* -- extension type registry ------------------------------------ */

        /**
         * The numeric id an extension name is written under.
         *
         * Ids are per sequence and assigned on first use, exactly as PyPulseq
         * assigns them, because the file declares the mapping itself: an
         * `extension ROTATIONS 3` line says what 3 means in *this* file.
         */
        int extension_type_id(const std::string& name);
        /** Look up without assigning; 0 if the name is not registered. */
        int find_extension_type_id(const std::string& name) const;
        /** The name an id was registered under, or empty. */
        const std::string& extension_type_name(int id) const;
        /** Force a name to a given id, as reading a file does. */
        void set_extension_type_id(const std::string& name, int id);

        /* -- label registry ---------------------------------------------- */
        /*
         * A label id means nothing on its own -- the file writes the *name*,
         * and the number is only how this sequence happens to have indexed it.
         * That matters because the numbering is not agreed anywhere: PyPulseq
         * knows 22 labels, the C reader's table knows 24, and a research
         * sequence may use a name neither has heard of.  So the sequence keeps
         * its own table, seeded with Pulseq's built-ins in Pulseq's order, and
         * a name outside them is appended.  Reading and writing both go
         * through the name, so the two never have to agree on a number.
         */

        /** Pulseq's built-in label names, in the order that numbers them. */
        static const std::vector<std::string>& builtin_labels();

        /** Id for @p name, appending it to this sequence's table if new. */
        int label_id(const std::string& name);
        /** Id for @p name without appending; 0 if unknown here. */
        int find_label_id(const std::string& name) const;
        /** The name id @p id was registered under, or empty. */
        const std::string& label_name(int id) const;
        /** Whether @p id is outside Pulseq's built-in set (so revision 1.5.2). */
        bool is_custom_label(int id) const;

        /* -- event registration ----------------------------------------- */
        /*
         * These append unconditionally rather than searching for an equal row.
         * A scan is built once and deduplicated once, at the end, over whole
         * columns -- searching per event would make building quadratic in the
         * thing that is already the largest.
         */

        int register_rf(const double* row, char use);
        int register_trap(const double* row);
        int register_arbitrary(const double* row);
        int register_adc(const double* row);
        int register_trigger(const double* row);
        int register_rotation(const double* row);
        int register_label_set(int32_t value, int32_t label_id);
        int register_label_inc(int32_t value, int32_t label_id);
        int register_rf_shim(const double* values, int count);
        int register_soft_delay(const SoftDelay& row);
        int register_shape(int num_uncompressed, const double* samples, int count);

        /**
         * Register a waveform as it stands, to be compressed before writing.
         *
         * See ShapeLibrary::append_raw.  A sequence built this way must reach
         * a writer (or `compress_shapes`) before its shapes mean what the file
         * format says they mean.
         */
        int register_raw_shape(const double* samples, int count);
        /** See ShapeLibrary::append_raw_divided. */
        int register_raw_shape_divided(const double* samples, int count, double divisor);

        /** Compress every shape still held raw.  Idempotent; writers call it. */
        void compress_shapes();

        /**
         * Append an extension-chain link.
         *
         * A chain is a singly-linked list through this library, so a block
         * carrying two extensions owns two rows of it.  Unlike the event
         * libraries this one is searched: chains are shared far more often
         * than they are distinct (every readout in a scan carries the same
         * pair of labels), and a scan-length library that deduplicates to
         * three rows is worth the lookup.
         */
        int chain_extension(int32_t type_id, int32_t ref, int32_t next);

        /**
         * Append a chain link without looking for an equal one.
         *
         * The counterpart to `chain_extension` for a caller that is going to
         * `remove_duplicates` anyway: the search is what makes chaining cost
         * more than appending, and a scan that labels every TR pays it a
         * million times to be told what the final pass would work out in one.
         *
         * Mixing the two is allowed and simply leaves duplicate links behind,
         * which deduplication collapses like any other repeated row.
         */
        int append_extension(int32_t type_id, int32_t ref, int32_t next);

        /* -- blocks ------------------------------------------------------ */

        int num_blocks() const
        {
            return static_cast<int>(durations_.size());
        }

        /** Append @p block.  @return its 1-based index. */
        int add_block(const Block& block);

        /** Overwrite block @p index (1-based).  Throws if out of range. */
        void set_block(int index, const Block& block);

        /** Block @p index (1-based).  Throws if out of range. */
        Block get_block(int index) const;

        /** Total playing time, the sum of the block durations. */
        double duration() const;

        /* -- deduplication ------------------------------------------------ */

        /**
         * Collapse every library onto its distinct rows and renumber what
         * points at them.  In place; the caller copies first if it wants both.
         *
         * `register_*` appends unconditionally, so a scan built block by block
         * arrives here holding one row per *use* of an event -- a readout
         * repeated a hundred thousand times is a hundred thousand identical
         * gradient rows.  This is the pass that makes it a file.
         *
         * Two rows are the same row when they agree at the precision the `.seq`
         * format writes that column at, which is why the surviving row is the
         * *rounded* one: keeping more precision than the file records would
         * mean the sequence read back differs from the one written.  See
         * dedup.cpp for the per-library profiles.
         */
        void remove_duplicates();

        /**
         * Whether every library is already down to its distinct rows.
         *
         * Set by @ref remove_duplicates and cleared by anything that could
         * undo it -- registering a row, editing the block table, or taking a
         * mutable reference to a library.  It is deliberately pessimistic: a
         * caller that asks for write access and then writes nothing loses the
         * claim, because the alternative is a claim that can be wrong, and a
         * wrong one is a file full of rows the writer was told it need not
         * collapse.
         *
         * Worth stating rather than re-deriving because deduplication is a
         * pass over every row of every library, and the ordinary path
         * deduplicates once and then writes.
         */
        bool deduplicated() const
        {
            return deduplicated_;
        }

        /**
         * Hand over a whole block table at once.
         *
         * A composed scan arrives as columns already -- that is what it was
         * built as -- so this takes them as they are rather than making the
         * caller replay millions of `add_block` calls to arrive back at the
         * same arrays.  @p events is row-major, BLOCK_WIDTH per block.
         */
        void set_blocks(const int32_t* events, const double* durations, int count);

        /** Raw block table, row-major, BLOCK_WIDTH per block. */
        const int32_t* block_events() const
        {
            return blocks_.data();
        }
        int32_t* block_events()
        {
            deduplicated_ = false;
            return blocks_.data();
        }
        const double* block_durations() const
        {
            return durations_.data();
        }
        double* block_durations()
        {
            deduplicated_ = false;
            return durations_.data();
        }

        /* -- gradients --------------------------------------------------- */

        /** What gradient id @p id is, and which row of which table holds it. */
        GradKind grad_kind(int id) const;
        /** Row index into the trapezoid or arbitrary table, 1-based. */
        int grad_row(int id) const;
        int num_gradients() const
        {
            return static_cast<int>(grad_slot_.size());
        }
        /** The signed slot: +row for a trapezoid, -row for an arbitrary. */
        const int32_t* grad_slots() const
        {
            return grad_slot_.data();
        }

        /* -- bulk loading ------------------------------------------------ */
        /*
         * A composed scan holds its libraries as dense arrays already, so these
         * take them as they are.  They replace rather than append, and they do
         * not check the ids they are given against the tables those ids point
         * into -- the caller built both.
         */

        /** Replace the gradient id -> signed slot map.  See the file comment. */
        void set_grad_slots(const int32_t* slots, int count);
        /** Replace the shape library.  @p starts holds @p count + 1 offsets. */
        void set_shapes(
            const int32_t* num_uncompressed,
            int count,
            const int32_t* starts,
            const double* samples);
        /** Replace the RF shim library.  @p starts holds @p count + 1 offsets. */
        void set_rf_shims(const int32_t* starts, int count, const double* values);

        /* -- libraries --------------------------------------------------- */

        const Table& rf_library() const
        {
            return rf_;
        }
        Table& rf_library()
        {
            deduplicated_ = false;
            return rf_;
        }
        const std::vector<char>& rf_uses() const
        {
            return rf_use_;
        }
        std::vector<char>& rf_uses()
        {
            deduplicated_ = false;
            return rf_use_;
        }

        const Table& trap_library() const
        {
            return trap_;
        }
        Table& trap_library()
        {
            deduplicated_ = false;
            return trap_;
        }
        const Table& arb_library() const
        {
            return arb_;
        }
        Table& arb_library()
        {
            deduplicated_ = false;
            return arb_;
        }
        const Table& adc_library() const
        {
            return adc_;
        }
        Table& adc_library()
        {
            deduplicated_ = false;
            return adc_;
        }
        const Table& trigger_library() const
        {
            return trigger_;
        }
        Table& trigger_library()
        {
            deduplicated_ = false;
            return trigger_;
        }
        const Table& rotation_library() const
        {
            return rotation_;
        }
        Table& rotation_library()
        {
            deduplicated_ = false;
            return rotation_;
        }

        const IntTable& extensions_library() const
        {
            return extensions_;
        }
        IntTable& extensions_library()
        {
            deduplicated_ = false;
            return extensions_;
        }
        const IntTable& label_set_library() const
        {
            return label_set_;
        }
        IntTable& label_set_library()
        {
            deduplicated_ = false;
            return label_set_;
        }
        const IntTable& label_inc_library() const
        {
            return label_inc_;
        }
        IntTable& label_inc_library()
        {
            deduplicated_ = false;
            return label_inc_;
        }

        const RaggedTable& rf_shim_library() const
        {
            return rf_shim_;
        }
        RaggedTable& rf_shim_library()
        {
            deduplicated_ = false;
            return rf_shim_;
        }
        const ShapeLibrary& shape_library() const
        {
            return shapes_;
        }
        ShapeLibrary& shape_library()
        {
            deduplicated_ = false;
            return shapes_;
        }
        const std::vector<SoftDelay>& soft_delay_library() const
        {
            return soft_delays_;
        }
        std::vector<SoftDelay>& soft_delay_library()
        {
            deduplicated_ = false;
            return soft_delays_;
        }

    private:
        int version_major_ = 1;
        int version_minor_ = 5;
        int version_revision_ = 0;

        double rf_raster_ = 1e-6;
        double grad_raster_ = 10e-6;
        double adc_raster_ = 100e-9;
        double block_raster_ = 10e-6;

        std::map<std::string, Definition> definitions_;

        std::map<std::string, int> extension_ids_;
        std::map<int, std::string> extension_names_;

        /** Label names by id-1; seeded with builtin_labels() and appended to. */
        std::vector<std::string> label_names_;
        std::map<std::string, int> label_ids_;

        Table rf_{RF_WIDTH};
        std::vector<char> rf_use_;
        Table trap_{TRAP_WIDTH};
        Table arb_{ARB_WIDTH};
        Table adc_{ADC_WIDTH};
        Table trigger_{TRIGGER_WIDTH};
        Table rotation_{ROTATION_WIDTH};
        IntTable extensions_{EXTENSION_WIDTH};
        IntTable label_set_{LABEL_WIDTH};
        IntTable label_inc_{LABEL_WIDTH};
        RaggedTable rf_shim_;
        ShapeLibrary shapes_;
        std::vector<SoftDelay> soft_delays_;

        /** grad id (1-based) -> +trap row / -arb row.  See the file comment. */
        std::vector<int32_t> grad_slot_;

        /** Extension chain rows by value, so a repeated chain costs one row. */
        std::map<std::array<int32_t, EXTENSION_WIDTH>, int> chain_index_;

        std::vector<int32_t> blocks_;
        std::vector<double> durations_;

        /** See deduplicated(); false until remove_duplicates() says otherwise. */
        bool deduplicated_ = false;
    };

} // namespace pulseq

#endif /* PULSEQ_CXX_SEQUENCE_HPP */
