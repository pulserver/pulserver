// Accelerators for fastseq.
//
// Everything here is optional and everything here has a NumPy or pure-Python
// twin in `fastseq.modules` that is the reference implementation, the fallback
// when this extension is not built, and what the tests compare against.
//
// `BlockWriter`, at the bottom, is the loop: one call per block of the scan.
// The rest are the two kernels deduplication spends its time in:
//
//   round_rows(table, digits) -> table rounded to the precision that decides
//       identity, column by column.  NumPy does this with about three strided
//       passes and a temporary per column, on top of a full copy; here the
//       table is read once, row-major, with every column resolved in registers.
//
//   unique_rows(rows) -> the distinct rows, in order of first appearance, and
//       which of them each row is.  NumPy's route is a sort of 64-bit digests,
//       which is O(n log n) and dominates the cost; an open-addressed table is
//       O(n) and can settle a hash collision on the spot, by comparing the rows
//       themselves, instead of falling back to a full lexicographic sort.
//
// Neither is allowed to differ from the Python version by a single bit: these
// numbers are what ends up in the .seq file, and identity of rows is what
// decides how many events the file has.  `tests/python/test_fastseq_dedup.py`
// checks both against it, including on the tables of a real MPRAGE.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

// What to do with a column: leave it alone, round it to a number of significant
// digits, or round it to a fixed number of decimal places.
enum class Mode { Copy, Significant, Fixed };

// Powers of ten at integer exponents.
//
// The scale a significant-digit column asks for is `10 ** (digit - ceil(log10
// |v|))`, whose exponent is always a whole number, so the `pow` NumPy does per
// element becomes a lookup -- and `pow` is the more expensive half of that
// column's arithmetic, the other being the `log10` itself.
//
// The table is filled with the same `std::pow` NumPy's `10.0 ** k` reaches, at
// the same integral arguments, so the two agree bit for bit; anything outside
// the range of a double falls back to `pow` and reaches the same infinity or
// zero it would have.
constexpr int kPowMin = -340;
constexpr int kPowMax = 340;

const double* power_table() {
  static const std::vector<double> table = [] {
    std::vector<double> values(static_cast<size_t>(kPowMax - kPowMin + 1));
    for (int k = kPowMin; k <= kPowMax; ++k) {
      values[static_cast<size_t>(k - kPowMin)] = std::pow(10.0, static_cast<double>(k));
    }
    return values;
  }();
  return table.data() - kPowMin;   // indexable by the exponent itself
}

inline double power_of_ten(double exponent, const double* table) {
  if (exponent >= static_cast<double>(kPowMin) && exponent <= static_cast<double>(kPowMax)) {
    return table[static_cast<int>(exponent)];
  }
  return std::pow(10.0, exponent);
}

// Rounding is worth about twenty per cent more when the compiler may use
// SSE4.1, because that is where `rint` stops being a call into libm and becomes
// a single instruction. Baking the requirement into the build would mean a
// wheel that crashes on a machine without it, so the kernel is cloned instead
// and the right version chosen when the module loads. Everywhere the attribute
// is unavailable -- other architectures, other compilers, macOS, which has no
// ifunc -- this expands to nothing and the baseline version runs.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(__APPLE__) && !defined(_WIN32)
#define FASTSEQ_MULTIVERSION __attribute__((target_clones("sse4.1", "default")))
#else
#define FASTSEQ_MULTIVERSION
#endif

// One value at the precision that decides identity. Both kernels below go
// through this, so the fused one cannot drift from the one the tests check.
inline double round_one(double v, Mode mode, double scale, const double* powers) {
  if (mode == Mode::Significant) {
    const double s = power_of_ten(scale - std::ceil(std::log10(std::fabs(v) + 1e-12)), powers);
    v = std::rint(v * s) / s;
  } else if (mode == Mode::Fixed) {
    v = std::rint(v * scale) / scale;
  }
  // `+= 0.0` in the Python, written out so that no optimiser is tempted to
  // decide it is a no-op. It is not: it turns -0.0 into 0.0.
  return (v == 0.0) ? 0.0 : v;
}

// The inner loop of `round_rows`, kept free of anything Python so that it can
// be cloned per instruction set and so that the GIL can stay released.
FASTSEQ_MULTIVERSION
void round_kernel(const double* src, double* dst, std::ptrdiff_t rows,
                  std::ptrdiff_t cols, const Mode* mode, const double* scale,
                  const double* powers) {
  for (std::ptrdiff_t r = 0; r < rows; ++r) {
    const double* in = src + r * cols;
    double* out = dst + r * cols;
    for (std::ptrdiff_t c = 0; c < cols; ++c) {
      out[c] = round_one(in[c], mode[c], scale[c], powers);
    }
  }
}

// The columns of `digits`, as the two kernels want them.
void rounding_plan(py::ssize_t cols, const std::vector<int>& digits,
                   std::vector<Mode>& mode, std::vector<double>& scale) {
  mode.assign(static_cast<size_t>(cols), Mode::Copy);
  scale.assign(static_cast<size_t>(cols), 1.0);
  const py::ssize_t named =
      std::min<py::ssize_t>(cols, static_cast<py::ssize_t>(digits.size()));
  for (py::ssize_t c = 0; c < named; ++c) {
    const int digit = digits[static_cast<size_t>(c)];
    if (digit > 0) {
      mode[static_cast<size_t>(c)] = Mode::Significant;
      scale[static_cast<size_t>(c)] = static_cast<double>(digit);
    } else {
      mode[static_cast<size_t>(c)] = Mode::Fixed;
      scale[static_cast<size_t>(c)] = std::pow(10.0, static_cast<double>(-digit));
    }
  }
}

// How many threads to split `n` rows across.
//
// Deduplicating and rendering are the two places left where a large scan spends
// real time on arithmetic rather than on memory, and both divide cleanly by
// rows. Splitting is worth a thread only when there is enough to divide, hence
// the floor; the ceiling is because what the chunks produce still has to be put
// back together in order, and they contend for the same memory long before
// twenty of them are useful. A short table -- which is every table in a
// sequence but these two -- takes the single-threaded path and spawns nothing.
unsigned split_threads(py::ssize_t n) {
  constexpr py::ssize_t least = 1 << 16;
  if (n < 2 * least) {
    return 1;
  }
  const unsigned cores = std::thread::hardware_concurrency();
  const unsigned wanted = static_cast<unsigned>(std::min<py::ssize_t>(n / least, 8));
  return std::max(1u, std::min(cores ? cores : 1u, wanted));
}

// Run `work(t)` for every chunk, this thread taking the first. Called with the
// GIL released and given work that never touches Python.
template <typename Work>
void run_split(unsigned threads, const Work& work) {
  if (threads <= 1) {
    work(0u);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(threads - 1);
  for (unsigned t = 1; t < threads; ++t) {
    pool.emplace_back(work, t);
  }
  work(0u);
  for (std::thread& worker : pool) {
    worker.join();
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// round_rows
// ---------------------------------------------------------------------------
//
// Mirrors, exactly:
//
//     out = table.copy()
//     for column, digit in enumerate(digits[: table.shape[1]]):
//         values = out[:, column]
//         if digit > 0:
//             scale = 10.0 ** (digit - np.ceil(np.log10(np.abs(values) + 1e-12)))
//         else:
//             scale = 10.0 ** (-digit)
//         out[:, column] = np.round(values * scale) / scale
//     out += 0.0
//
// `np.round` with no decimals is `np.rint`, which rounds halves to even; so
// does C's `rint` under the default rounding mode, which is what makes the two
// agree on the ties.  Columns past the end of `digits` are copied untouched,
// and every column has zero added to it, which is the identity everywhere
// except on negative zero -- and negative zero is a row of its own as far as a
// bitwise comparison is concerned, so folding it matters.
static py::array_t<double> round_rows(
    py::array_t<double, py::array::c_style | py::array::forcecast> table,
    const std::vector<int>& digits) {
  const py::buffer_info info = table.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("round_rows expects a two-dimensional table");
  }
  const py::ssize_t rows = info.shape[0];
  const py::ssize_t cols = info.shape[1];

  py::array_t<double> out({rows, cols});
  if (rows == 0 || cols == 0) {
    return out;
  }

  std::vector<Mode> mode;
  std::vector<double> scale;
  rounding_plan(cols, digits, mode, scale);

  const double* src = static_cast<const double*>(info.ptr);
  double* dst = out.mutable_data();
  const double* powers = power_table();

  {
    py::gil_scoped_release unlocked;
    round_kernel(src, dst, rows, cols, mode.data(), scale.data(), powers);
  }
  return out;
}

// ---------------------------------------------------------------------------
// unique_rows
// ---------------------------------------------------------------------------
//
// Rows are compared as bit patterns, which is what the Python version's
// verification step does too, and works for any eight-byte element type -- the
// float tables of the event libraries and the integer keys of the extension
// chains both come through here.
//
// The hash is FNV-1a over the row's words with a final avalanche, because FNV's
// low bits -- the ones an open-addressed table indexes with -- are weak on
// inputs that share a suffix, which whole columns of these tables do.  It does
// not have to agree with the Python version's hash: a hash decides only where
// probing starts, and every apparent hit is confirmed against the row itself.
static py::tuple unique_rows(py::array rows) {
  const py::buffer_info info = rows.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("unique_rows expects a two-dimensional table");
  }
  if (info.itemsize != 8) {
    throw std::invalid_argument("unique_rows expects an eight-byte element type");
  }
  const py::ssize_t n = info.shape[0];
  const py::ssize_t m = info.shape[1];
  if (info.strides[1] != 8 || info.strides[0] != m * 8) {
    throw std::invalid_argument("unique_rows expects a C-contiguous table");
  }

  py::array_t<int64_t> code(n);
  if (n == 0) {
    py::array empty(rows.dtype(), std::vector<py::ssize_t>{0, m});
    return py::make_tuple(empty, code);
  }

  const uint64_t* words = static_cast<const uint64_t*>(info.ptr);
  int64_t* codes = code.mutable_data();

  const auto digest = [&](const uint64_t* row) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (py::ssize_t c = 0; c < m; ++c) {
      h ^= row[c];
      h *= 0x100000001b3ULL;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
  };

  // Add a row to a table of distinct ones, keyed by what it holds, and say
  // which one it is. `first` indexes into the table being deduplicated, so a row
  // is never copied to be compared.
  const auto claim = [&](std::vector<int64_t>& slot, size_t mask,
                         std::vector<py::ssize_t>& first, py::ssize_t r) {
    const uint64_t* row = words + r * m;
    size_t at = static_cast<size_t>(digest(row)) & mask;
    for (;;) {
      const int64_t held = slot[at];
      if (held < 0) {
        const int64_t issued = static_cast<int64_t>(first.size());
        slot[at] = issued;
        first.push_back(r);
        return issued;
      }
      const uint64_t* candidate = words + first[static_cast<size_t>(held)] * m;
      if (std::memcmp(candidate, row, static_cast<size_t>(m) * 8) == 0) {
        return held;
      }
      at = (at + 1) & mask;
    }
  };

  // Load factor one half, so probing stays short even when every row is unique.
  const auto sized_for = [](py::ssize_t span) {
    size_t capacity = 16;
    while (capacity < static_cast<size_t>(span) * 2) {
      capacity <<= 1;
    }
    return capacity;
  };

  // A chunk's distinct rows, as indices into the whole table, with the codes it
  // wrote being local to it. Everything is the same as one pass would do except
  // that the codes have to be translated once the chunks are merged.
  const auto distinct_within = [&](py::ssize_t from, py::ssize_t upto,
                                   std::vector<py::ssize_t>& first) {
    const size_t capacity = sized_for(upto - from);
    std::vector<int64_t> slot(capacity, -1);
    first.reserve(static_cast<size_t>(upto - from) / 8 + 16);
    for (py::ssize_t r = from; r < upto; ++r) {
      codes[r] = claim(slot, capacity - 1, first, r);
    }
  };

  std::vector<py::ssize_t> first;
  const unsigned threads = split_threads(n);
  {
    py::gil_scoped_release unlocked;
    if (threads == 1) {
      distinct_within(0, n, first);
    } else {
      std::vector<py::ssize_t> edge(threads + 1);
      for (unsigned t = 0; t <= threads; ++t) {
        edge[t] = n * t / threads;
      }

      std::vector<std::vector<py::ssize_t>> found(threads);
      run_split(threads, [&](unsigned t) {
        distinct_within(edge[t], edge[t + 1], found[t]);
      });

      // Merged in chunk order, so a row's code is decided by where it first
      // turns up in the whole table -- which is what a single pass would have
      // said, and why the answer does not depend on how many threads ran.
      const size_t capacity = sized_for(n);
      std::vector<int64_t> slot(capacity, -1);
      std::vector<std::vector<int64_t>> renumber(threads);
      for (unsigned t = 0; t < threads; ++t) {
        renumber[t].reserve(found[t].size());
        for (py::ssize_t candidate : found[t]) {
          renumber[t].push_back(claim(slot, capacity - 1, first, candidate));
        }
      }

      run_split(threads, [&](unsigned t) {
        const int64_t* mapping = renumber[t].data();
        for (py::ssize_t r = edge[t]; r < edge[t + 1]; ++r) {
          codes[r] = mapping[codes[r]];
        }
      });
    }
  }

  const py::ssize_t kept = static_cast<py::ssize_t>(first.size());
  py::array unique(rows.dtype(), std::vector<py::ssize_t>{kept, m});
  uint64_t* out = static_cast<uint64_t*>(unique.request().ptr);
  {
    py::gil_scoped_release unlocked;
    for (py::ssize_t u = 0; u < kept; ++u) {
      std::memcpy(out + u * m, words + first[static_cast<size_t>(u)] * m,
                  static_cast<size_t>(m) * 8);
    }
  }
  return py::make_tuple(unique, code);
}

// ---------------------------------------------------------------------------
// collapse_rows
// ---------------------------------------------------------------------------
//
// Mirrors, exactly:
//
//     rows = _rounded(table[take], digits)
//     keyed = rows if extra is None else np.column_stack([rows, extra[take]])
//     unique, code = _unique_rows(keyed)
//
// which is what deduplicating an event library is, and which in NumPy builds
// the whole gathered and rounded table first: on a large 3D scan that is four
// million rows of six doubles, two hundred megabytes written and then read
// twice, to end up with a couple of thousand distinct rows.  Nothing here needs
// that table to exist -- a row is rounded, hashed and thrown away in the same
// breath -- so it never does.
//
// The other half of the saving is the hash table.  `unique_rows` cannot know
// how many distinct rows it will find and sizes its table for the worst case,
// which on this input is sixty-four megabytes of mostly empty slots; here the
// table grows from small, so it stays in cache for the whole pass.
//
// `extra` is keyed on without being rounded: an RF pulse's `use` decides
// identity but is not part of its data.

namespace {

struct Collapsed {
  std::vector<double> unique;   // distinct rows, in order of first appearance
  py::ssize_t width = 0;
};

inline uint64_t hash_words(const uint64_t* words, py::ssize_t width) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (py::ssize_t c = 0; c < width; ++c) {
    h ^= words[c];
    h *= 0x100000001b3ULL;
  }
  h ^= h >> 33;
  return h;
}

FASTSEQ_MULTIVERSION
void collapse_kernel(const double* table, py::ssize_t cols, const int64_t* take,
                     py::ssize_t n, const Mode* mode, const double* scale,
                     const double* powers, const double* extra, py::ssize_t width,
                     Collapsed& out, int64_t* code) {
  std::vector<int64_t> slot(1024, -1);
  size_t mask = slot.size() - 1;
  std::vector<double> key(static_cast<size_t>(width));

  for (py::ssize_t r = 0; r < n; ++r) {
    const double* in = table + take[r] * cols;
    for (py::ssize_t c = 0; c < cols; ++c) {
      key[static_cast<size_t>(c)] = round_one(in[c], mode[c], scale[c], powers);
    }
    if (extra != nullptr) {
      key[static_cast<size_t>(cols)] = extra[take[r]];
    }

    const uint64_t* words = reinterpret_cast<const uint64_t*>(key.data());
    size_t at = static_cast<size_t>(hash_words(words, width)) & mask;
    int64_t found = -1;
    while (slot[at] >= 0) {
      const double* candidate = out.unique.data() + slot[at] * width;
      if (std::memcmp(candidate, key.data(), static_cast<size_t>(width) * 8) == 0) {
        found = slot[at];
        break;
      }
      at = (at + 1) & mask;
    }
    if (found < 0) {
      found = static_cast<int64_t>(out.unique.size() / static_cast<size_t>(width));
      out.unique.insert(out.unique.end(), key.begin(), key.end());
      slot[at] = found;

      // Load factor one half, as everywhere else here. Distinct rows are few
      // in every sequence that has ever been built, so this rarely runs twice.
      if ((out.unique.size() / static_cast<size_t>(width)) * 2 > slot.size()) {
        std::vector<int64_t> grown(slot.size() * 2, -1);
        const size_t grown_mask = grown.size() - 1;
        const py::ssize_t kept =
            static_cast<py::ssize_t>(out.unique.size() / static_cast<size_t>(width));
        for (py::ssize_t u = 0; u < kept; ++u) {
          const double* row = out.unique.data() + u * width;
          size_t to = static_cast<size_t>(
                          hash_words(reinterpret_cast<const uint64_t*>(row), width)) &
                      grown_mask;
          while (grown[to] >= 0) {
            to = (to + 1) & grown_mask;
          }
          grown[to] = u;
        }
        slot.swap(grown);
        mask = grown_mask;
      }
    }
    code[r] = found;
  }
}

// An open-addressed index over a table of already-rounded rows, which hands out
// an ID per distinct row in the order it first sees one. The same table and the
// same hashing as `collapse_kernel`, without the rounding: this is for merging
// results that have already been through it.
struct RowIndex {
  py::ssize_t width;
  std::vector<double>* rows;
  std::vector<int64_t> slot = std::vector<int64_t>(1024, -1);
  size_t mask = 1023;

  int64_t insert(const double* key) {
    size_t at = static_cast<size_t>(hash_words(reinterpret_cast<const uint64_t*>(key),
                                               width)) & mask;
    while (slot[at] >= 0) {
      if (std::memcmp(rows->data() + slot[at] * width, key,
                      static_cast<size_t>(width) * 8) == 0) {
        return slot[at];
      }
      at = (at + 1) & mask;
    }
    const int64_t issued =
        static_cast<int64_t>(rows->size() / static_cast<size_t>(width));
    rows->insert(rows->end(), key, key + width);
    slot[at] = issued;
    if ((rows->size() / static_cast<size_t>(width)) * 2 > slot.size()) {
      std::vector<int64_t> grown(slot.size() * 2, -1);
      const size_t grown_mask = grown.size() - 1;
      const py::ssize_t kept =
          static_cast<py::ssize_t>(rows->size() / static_cast<size_t>(width));
      for (py::ssize_t u = 0; u < kept; ++u) {
        const double* row = rows->data() + u * width;
        size_t to = static_cast<size_t>(
                        hash_words(reinterpret_cast<const uint64_t*>(row), width)) &
                    grown_mask;
        while (grown[to] >= 0) {
          to = (to + 1) & grown_mask;
        }
        grown[to] = u;
      }
      slot.swap(grown);
      mask = grown_mask;
    }
    return issued;
  }
};

}  // namespace

static py::tuple collapse_rows(
    py::array_t<double, py::array::c_style | py::array::forcecast> table,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> take,
    const std::vector<int>& digits, py::object extra_obj) {
  const py::buffer_info info = table.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("collapse_rows expects a two-dimensional table");
  }
  const py::ssize_t cols = info.shape[1];
  const py::ssize_t rows = info.shape[0];
  const py::buffer_info picks = take.request();
  if (picks.ndim != 1) {
    throw std::invalid_argument("collapse_rows expects a one-dimensional selection");
  }
  const py::ssize_t n = picks.shape[0];

  py::array_t<double> extra_array;
  const double* extra = nullptr;
  if (!extra_obj.is_none()) {
    extra_array = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
        extra_obj);
    if (extra_array.request().shape[0] != rows) {
      throw std::invalid_argument("collapse_rows expects one extra value per table row");
    }
    extra = extra_array.data();
  }
  const py::ssize_t width = cols + (extra != nullptr ? 1 : 0);

  py::array_t<int64_t> code(n);
  if (n == 0 || width == 0) {
    const std::vector<py::ssize_t> empty{0, width};
    return py::make_tuple(py::array_t<double>(empty), code);
  }

  const int64_t* picked = static_cast<const int64_t*>(picks.ptr);
  for (py::ssize_t r = 0; r < n; ++r) {
    if (picked[r] < 0 || picked[r] >= rows) {
      throw std::out_of_range("collapse_rows was given a row that is not in the table");
    }
  }

  std::vector<Mode> mode;
  std::vector<double> scale;
  rounding_plan(cols, digits, mode, scale);

  const double* values = static_cast<const double*>(info.ptr);
  int64_t* codes = code.mutable_data();
  const unsigned threads = split_threads(n);

  Collapsed out;
  out.width = width;
  {
    py::gil_scoped_release unlocked;
    if (threads == 1) {
      collapse_kernel(values, cols, picked, n, mode.data(), scale.data(), power_table(),
                      extra, width, out, codes);
    } else {
      // Each chunk collapses against a table of its own, so the codes it writes
      // are local to it and have to be translated afterwards.
      std::vector<py::ssize_t> edge(threads + 1);
      for (unsigned t = 0; t <= threads; ++t) {
        edge[t] = n * t / threads;
      }

      std::vector<Collapsed> found(threads);
      run_split(threads, [&](unsigned t) {
        found[t].width = width;
        collapse_kernel(values, cols, picked + edge[t], edge[t + 1] - edge[t],
                        mode.data(), scale.data(), power_table(), extra, width,
                        found[t], codes + edge[t]);
      });

      // Merged in chunk order, so a row's ID is decided by where it first turns
      // up in the whole selection -- which is what a single pass would have
      // said, and why the answer does not depend on how many threads ran.
      RowIndex index{width, &out.unique};
      std::vector<std::vector<int64_t>> renumber(threads);
      for (unsigned t = 0; t < threads; ++t) {
        const py::ssize_t mine =
            static_cast<py::ssize_t>(found[t].unique.size() / static_cast<size_t>(width));
        renumber[t].reserve(static_cast<size_t>(mine));
        for (py::ssize_t u = 0; u < mine; ++u) {
          renumber[t].push_back(index.insert(found[t].unique.data() + u * width));
        }
        std::vector<double>().swap(found[t].unique);
      }

      run_split(threads, [&](unsigned t) {
        const int64_t* mapping = renumber[t].data();
        for (py::ssize_t r = edge[t]; r < edge[t + 1]; ++r) {
          codes[r] = mapping[codes[r]];
        }
      });
    }
  }

  const py::ssize_t kept =
      static_cast<py::ssize_t>(out.unique.size() / static_cast<size_t>(width));
  py::array_t<double> unique({kept, width});
  std::memcpy(unique.mutable_data(), out.unique.data(),
              out.unique.size() * sizeof(double));
  return py::make_tuple(unique, code);
}

// ---------------------------------------------------------------------------
// render_int_rows
// ---------------------------------------------------------------------------
//
// `[BLOCKS]` as text.  It is one row per block rather than per distinct event,
// so on a large 3D protocol it is three quarters of the file, and rendering it
// is the largest single cost of writing one.
//
// `_render_int_rows` in `fastseq/sequence.py` is the reference: it does this
// with whole-column NumPy arithmetic into a padded character matrix and then a
// boolean gather to drop the padding, which reads and writes some hundred and
// thirty megabytes of temporaries for a sixty megabyte section.  Here each
// value's digits are written where they belong the first time.
//
// Both produce exactly what `' '.join('%{width}d' % v) + '\n'` would, per row,
// and `None` when a value is negative -- `%d` widens rather than truncating, so
// there is no width to fall back on and the caller formats those rows itself.

static py::object render_int_rows(py::array matrix, const std::vector<int>& widths) {
  py::buffer_info info = matrix.request();
  if (info.ndim != 2 || info.itemsize != 8) {
    throw std::invalid_argument("render_int_rows expects a two-dimensional 8-byte table");
  }
  const py::ssize_t n = info.shape[0];
  const py::ssize_t m = info.shape[1];
  if (info.strides[1] != 8 || info.strides[0] != m * 8) {
    throw std::invalid_argument("render_int_rows expects a C-contiguous table");
  }
  if (static_cast<size_t>(m) != widths.size() || n == 0) {
    return py::none();
  }

  const int64_t* values = static_cast<const int64_t*>(info.ptr);

  // Rows are independent, so both passes below are split the same way. The
  // chunks are contiguous and rendered into buffers of their own, which is what
  // makes them independent -- a row's rendered length depends on its digits, so
  // there is no way to know where a row lands without rendering everything
  // before it.
  const unsigned threads = split_threads(n);
  std::vector<py::ssize_t> edge(threads + 1);
  for (unsigned t = 0; t <= threads; ++t) {
    edge[t] = n * t / threads;
  }

  // The widest each column gets, which is what bounds the buffer. A negative
  // anywhere means this cannot be rendered at a fixed width at all.
  std::vector<int> span(static_cast<size_t>(m));
  {
    py::gil_scoped_release unlocked;
    std::vector<std::vector<int64_t>> found(threads);
    const auto scan = [&](unsigned t) {
      std::vector<int64_t>& maxima = found[t];
      maxima.assign(static_cast<size_t>(m), 0);
      for (py::ssize_t r = edge[t]; r < edge[t + 1]; ++r) {
        const int64_t* row = values + r * m;
        for (py::ssize_t c = 0; c < m; ++c) {
          if (row[c] < 0) {
            maxima[0] = -1;
            return;
          }
          if (row[c] > maxima[static_cast<size_t>(c)]) {
            maxima[static_cast<size_t>(c)] = row[c];
          }
        }
      }
    };
    run_split(threads, scan);

    std::vector<int64_t> maxima(static_cast<size_t>(m), 0);
    for (unsigned t = 0; t < threads; ++t) {
      if (found[t][0] < 0) {
        maxima[0] = -1;
        break;
      }
      for (py::ssize_t c = 0; c < m; ++c) {
        maxima[static_cast<size_t>(c)] =
            std::max(maxima[static_cast<size_t>(c)], found[t][static_cast<size_t>(c)]);
      }
    }
    if (maxima[0] < 0) {
      span.clear();
    } else {
      for (py::ssize_t c = 0; c < m; ++c) {
        int digits = 1;
        for (int64_t limit = maxima[static_cast<size_t>(c)]; limit >= 10; limit /= 10) {
          ++digits;
        }
        span[static_cast<size_t>(c)] = std::max(widths[static_cast<size_t>(c)], digits);
      }
    }
  }
  if (span.empty()) {
    return py::none();
  }

  size_t line = 0;
  for (py::ssize_t c = 0; c < m; ++c) {
    line += static_cast<size_t>(span[static_cast<size_t>(c)]) + 1;  // separator, newline last
  }

  std::string out;
  {
    py::gil_scoped_release unlocked;
    std::vector<std::string> part(threads);
    const auto render = [&](unsigned t) {
      std::string& buffer = part[t];
      // An upper bound: rows whose values are narrower than the column shrink
      // it, which is what the resize at the end of the chunk settles.
      buffer.resize(static_cast<size_t>(edge[t + 1] - edge[t]) * line);
      char* at = buffer.data();
      size_t used = 0;
      char digits[24];
      for (py::ssize_t r = edge[t]; r < edge[t + 1]; ++r) {
        const int64_t* row = values + r * m;
        for (py::ssize_t c = 0; c < m; ++c) {
          int64_t value = row[c];
          int count = 0;
          do {
            digits[count++] = static_cast<char>('0' + value % 10);
            value /= 10;
          } while (value != 0);
          for (int pad = widths[static_cast<size_t>(c)] - count; pad > 0; --pad) {
            at[used++] = ' ';
          }
          while (count > 0) {
            at[used++] = digits[--count];
          }
          at[used++] = (c + 1 == m) ? '\n' : ' ';
        }
      }
      buffer.resize(used);
    };
    run_split(threads, render);

    if (threads == 1) {
      out.swap(part[0]);
    } else {
      size_t total = 0;
      std::vector<size_t> start(threads);
      for (unsigned t = 0; t < threads; ++t) {
        start[t] = total;
        total += part[t].size();
      }
      out.resize(total);
      char* buffer = out.data();
      run_split(threads, [&](unsigned t) {
        std::memcpy(buffer + start[t], part[t].data(), part[t].size());
      });
    }
  }
  // Bytes rather than text: this is ASCII on its way to a file, and handing it
  // back as `str` costs a decode here and an encode again at the end -- two
  // passes over most of a large .seq file, to arrive at the bytes we had.
  return py::bytes(out);
}

// ---------------------------------------------------------------------------
// tile_rows
// ---------------------------------------------------------------------------

// `period` laid out `loop_size` times, end to end.
//
// What `np.tile` does, on threads. The period is a few thousand rows and stays
// in cache; the destination is the whole scan -- eighty megabytes of trapezoid
// rows on a large 3D protocol -- so this is a pure streaming write, and a
// streaming write is bounded by how much memory traffic one core can ask for
// rather than by anything it computes.
py::array_t<double> tile_rows(py::array_t<double, py::array::c_style> period,
                              py::ssize_t loop_size) {
  py::buffer_info info = period.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("tile_rows expects a two-dimensional table");
  }
  if (loop_size < 0) {
    throw std::invalid_argument("tile_rows expects a non-negative repeat count");
  }
  const py::ssize_t rows = info.shape[0];
  const py::ssize_t cols = info.shape[1];

  py::array_t<double> out({rows * loop_size, cols});
  if (rows == 0 || cols == 0 || loop_size == 0) {
    return out;
  }

  const double* source = static_cast<const double*>(info.ptr);
  double* target = out.mutable_data();
  const size_t span = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  {
    py::gil_scoped_release unlocked;
    const unsigned threads = split_threads(rows * loop_size);
    run_split(threads, [&](unsigned t) {
      const py::ssize_t from = loop_size * t / threads;
      const py::ssize_t upto = loop_size * (t + 1) / threads;
      for (py::ssize_t copy = from; copy < upto; ++copy) {
        std::memcpy(target + static_cast<size_t>(copy) * span, source, span * sizeof(double));
      }
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Block columns
// ---------------------------------------------------------------------------

// Turn the loop's slot offsets and switches into the columns a block table is
// made of, plus the index its gradient column points through.
//
// Every one of these is a rank: the dense tables are laid out in block order, so
// a row's library ID is just how many rows of its kind come before it, and a
// slot that was allocated but never switched on contributes a rank that nothing
// refers to -- which is exactly what lets pruning drop it later.
//
// Done here rather than in NumPy because there it is six separate passes over
// arrays three times the length of the scan -- a running sum, two comparisons, a
// gather and two strided writes into a seven-column table -- each of them
// reading and writing tens of megabytes to carry a few bits of state forward.
// The whole derivation is one sequential walk with one counter in it, and that
// is what this is. `_block_columns_py` in `fastseq/modules.py` is the same
// derivation in NumPy, and the two are compared row for row in the tests.
py::tuple block_columns(py::array_t<int32_t, py::array::c_style> slots, py::object on,
                        int rf_width, int adc_width, int trap_width, int arb_width) {
  py::buffer_info info = slots.request();
  if (info.ndim != 2 || info.shape[1] != 5) {
    throw std::invalid_argument("block_columns expects a five-column slot table");
  }
  if (!PyByteArray_Check(on.ptr())) {
    throw std::invalid_argument("block_columns expects the switches as a bytearray");
  }
  const py::ssize_t blocks = info.shape[0];
  if (PyByteArray_GET_SIZE(on.ptr()) != 5 * blocks) {
    throw std::invalid_argument("block_columns: the switches do not match the slot table");
  }

  const int32_t* slot = static_cast<const int32_t*>(info.ptr);
  const unsigned char* switches =
      reinterpret_cast<const unsigned char*>(PyByteArray_AS_STRING(on.ptr()));

  // How many gradient slots were played, which is how many rows the index has.
  // Counting first costs one sequential read and saves either a guess at the
  // size or a copy out of a vector that grew to three times the scan.
  py::ssize_t sounding = 0;
  {
    py::gil_scoped_release unlocked;
    for (py::ssize_t i = 0; i < blocks; ++i) {
      const int32_t* here = slot + 5 * i;
      const unsigned char* played = switches + 5 * i;
      for (int axis = 0; axis < 3; ++axis) {
        sounding += (here[1 + axis] != 0) && played[1 + axis];
      }
    }
  }

  py::array_t<int32_t> rf_column(blocks);
  py::array_t<int32_t> adc_column(blocks);
  py::array_t<int32_t> grad_column(3 * blocks);
  py::array_t<int32_t> index({sounding, py::ssize_t{2}});

  int32_t* rf = rf_column.mutable_data();
  int32_t* adc = adc_column.mutable_data();
  int32_t* grad = grad_column.mutable_data();
  int32_t* pointed = index.mutable_data();

  {
    py::gil_scoped_release unlocked;
    int32_t rank = 0;
    py::ssize_t at = 0;
    for (py::ssize_t i = 0; i < blocks; ++i) {
      const int32_t* here = slot + 5 * i;
      const unsigned char* played = switches + 5 * i;

      // A slot of zero is "no such event" and stays zero. Saying so costs a
      // branch and buys agreement with the NumPy twin, which arrives at the
      // same answer by flooring a negative division rather than by asking.
      rf[i] = (played[0] && here[0] != 0) ? (here[0] - 1) / rf_width + 1 : 0;
      adc[i] = (played[4] && here[4] != 0) ? (here[4] - 1) / adc_width + 1 : 0;

      for (int axis = 0; axis < 3; ++axis) {
        const int32_t offset = here[1 + axis];
        if (offset == 0) {
          grad[3 * i + axis] = 0;
          continue;
        }
        // The rank advances whether or not the block played it: it numbers the
        // rows that exist, which is what makes a skipped one free.
        ++rank;
        if (!played[1 + axis]) {
          grad[3 * i + axis] = 0;
          continue;
        }
        grad[3 * i + axis] = rank;

        // Only what was played is indexed. Deduplication reads these rows
        // straight through, in this order, so indexing every allocated slot
        // instead would mean gathering the played ones back out of it -- a
        // scan-length pass to arrive at the order they were already in.
        // The sign of the offset is which of the two tables it points into:
        // positive is a trapezoid, negative an arbitrary waveform.
        const bool arbitrary = offset < 0;
        pointed[2 * at] = arbitrary;
        pointed[2 * at + 1] =
            ((arbitrary ? -offset : offset) - 1) / (arbitrary ? arb_width : trap_width);
        ++at;
      }
    }
  }
  return py::make_tuple(rf_column, grad_column, adc_column, index);
}

// ---------------------------------------------------------------------------
// BlockWriter
// ---------------------------------------------------------------------------
//
// `PeriodicSequence.add_block`, which runs once per block and is where a large
// scan spends most of its design time. `_add_block_py` in `fastseq/modules.py`
// is the same method in Python, is what this must agree with down to the
// wording of its errors, and is what runs when this extension is absent.
//
// What it does is not much: read a few attributes off each event handed in, put
// the numbers in the row the constructor already set aside for this block, and
// mark the slot as played. What made it expensive in Python is that the two
// halves of that -- an attribute read and a store through a `memoryview` --
// cost about twenty-four nanoseconds each, and a four-event block does twenty
// of them. Here the stores are stores; the attribute reads are the floor, and
// remain the largest part of what is left.
//
// The writer holds raw pointers into buffers the constructor allocated and
// nothing afterwards reallocates. It also holds a reference to each of them, so
// none can be collected while it is alive. The bytearrays are the exception:
// their contents can in principle move, so their pointer is taken fresh on
// every call, which is one load.

namespace {

// Attribute names, interned once. `PyObject_GetAttr` on an interned string
// takes the dictionary fast path; building the string per access would not.
struct Names {
  py::object type, channel, amplitude, first, last, delay, duration, value, label;
  py::object freq_ppm, phase_ppm, freq_offset, phase_offset;
  py::object ps_code, ps_shape, rot_quaternion, as_quat;

  static py::object intern(const char* text) {
    return py::reinterpret_steal<py::object>(PyUnicode_InternFromString(text));
  }

  Names()
      : type(intern("type")),
        channel(intern("channel")),
        amplitude(intern("amplitude")),
        first(intern("first")),
        last(intern("last")),
        delay(intern("delay")),
        duration(intern("duration")),
        value(intern("value")),
        label(intern("label")),
        freq_ppm(intern("freq_ppm")),
        phase_ppm(intern("phase_ppm")),
        freq_offset(intern("freq_offset")),
        phase_offset(intern("phase_offset")),
        ps_code(intern("_ps_code")),
        ps_shape(intern("_ps_shape")),
        rot_quaternion(intern("rot_quaternion")),
        as_quat(intern("as_quat")) {}
};

const Names& names() {
  static const Names* n = new Names();   // never destroyed: outlives the interpreter
  return *n;
}

// `types.SimpleNamespace`, which is what every event PyPulseq builds is.
PyTypeObject* namespace_type() {
  static PyTypeObject* type = reinterpret_cast<PyTypeObject*>(
      py::object(py::module_::import("types").attr("SimpleNamespace")).release().ptr());
  return type;
}

// The instance dictionary of an event, when reading it straight is allowed.
//
// A block reads a handful of attributes off each event handed to it, and those
// reads are what the loop costs -- `PyObject_GetAttr` walks the type's MRO
// looking for a descriptor before it ever reaches the instance dictionary, and
// on a `SimpleNamespace`, which defines none, that walk can only come back with
// what the dictionary already had.
//
// Only an exact `SimpleNamespace` takes this path. A subclass may define a
// descriptor that has to win over the dictionary, and a `Rotation` is not one of
// these at all; both go the ordinary way, as does an attribute this does not
// find, so what an event resolves to is unchanged in every case.
inline PyObject* namespace_dict(PyObject* obj) {
  if (Py_TYPE(obj) != namespace_type()) {
    return nullptr;
  }
  const Py_ssize_t offset = Py_TYPE(obj)->tp_dictoffset;
  if (offset <= 0) {
    return nullptr;
  }
  return *reinterpret_cast<PyObject**>(reinterpret_cast<char*>(obj) + offset);
}

// A missing attribute is not an error here -- it is how "this event has no code
// yet" and "this waveform was copied before the sequence was composed" are
// asked about -- so the exception is swallowed and the caller decides.
inline PyObject* try_getattr(PyObject* obj, const py::object& name) {
  PyObject* dict = namespace_dict(obj);
  if (dict != nullptr) {
    PyObject* found = PyDict_GetItemWithError(dict, name.ptr());   // borrowed
    if (found != nullptr) {
      Py_INCREF(found);
      return found;
    }
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return nullptr;
    }
  }
  PyObject* value = PyObject_GetAttr(obj, name.ptr());
  if (value == nullptr) {
    PyErr_Clear();
  }
  return value;
}

inline double as_double(PyObject* value) {
  if (PyFloat_CheckExact(value)) {
    return PyFloat_AS_DOUBLE(value);
  }
  const double out = PyFloat_AsDouble(value);
  if (out == -1.0 && PyErr_Occurred()) {
    throw py::error_already_set();
  }
  return out;
}

// Read one float attribute. The event types that reach here are PyPulseq's
// SimpleNamespaces, whose attributes are all present; a missing one is a
// genuine error and is raised as one.
inline double float_attr(PyObject* obj, const py::object& name) {
  PyObject* dict = namespace_dict(obj);
  if (dict != nullptr) {
    PyObject* found = PyDict_GetItemWithError(dict, name.ptr());   // borrowed
    if (found != nullptr) {
      return as_double(found);
    }
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
  }
  PyObject* value = PyObject_GetAttr(obj, name.ptr());
  if (value == nullptr) {
    throw py::error_already_set();
  }
  const double out = as_double(value);
  Py_DECREF(value);
  return out;
}

// A flat, writable buffer of `itemsize` elements behind a memoryview or array.
struct Flat {
  void* data = nullptr;
  py::ssize_t count = 0;
};

Flat flat_buffer(const py::object& source, char kind, const char* what) {
  if (source.is_none()) {
    return Flat{};
  }
  py::buffer_info info = py::cast<py::buffer>(source).request(true);
  const py::ssize_t want = (kind == 'd') ? 8 : 4;
  if (info.itemsize != want || info.format.empty() || info.format[0] != kind) {
    throw std::invalid_argument(std::string("BlockWriter: ") + what +
                                " is not the expected element type");
  }
  py::ssize_t count = 1;
  for (py::ssize_t n : info.shape) {
    count *= n;
  }
  return Flat{info.ptr, count};
}

PyObject* checked_bytearray(const py::object& source, const char* what) {
  if (!PyByteArray_Check(source.ptr())) {
    throw std::invalid_argument(std::string("BlockWriter: ") + what +
                                " is not a bytearray");
  }
  return source.ptr();
}

}  // namespace

class BlockWriter {
 public:
  // `owner` is read here and not kept. It used to be a member, along with its
  // instance dict, purely so that `cursor` could be written back into it after
  // every block -- and that back-reference made the sequence immortal: the
  // sequence holds this writer, this writer held the sequence, and a pybind11
  // class has no `tp_traverse`, so Python's cycle collector cannot see through
  // it to break the loop. A large scan then leaks its dense tables, about a
  // gigabyte of them, on every build. `cursor` is read back out of this object
  // instead, which is both cheaper and severable.
  explicit BlockWriter(const py::object& owner) {
    // The module the sequence's own class came from, rather than a name spelled
    // out here. It is reached on the same terms as everything else in this
    // package -- as part of `fastseq` when there is one, and on its own when the
    // directory itself is what is on the path, which is what
    // `cd fastseq; import modules` gives -- and asking the owner is how this
    // side finds out which of the two happened. The refusal helpers and the
    // label tables live over there either way.
    py::module_ modules = py::module_::import(
        py::cast<std::string>(py::getattr(owner, "__module__")).c_str());
    no_slot_ = modules.attr("_no_slot");
    no_extension_ = modules.attr("_no_extension");
    stale_copy_ = modules.attr("_stale_copy");
    wrong_shim_ = modules.attr("_wrong_shim");
    adopt_ = modules.attr("_adopt");
    trigger_type_ = modules.attr("_TRIGGER_TYPE");
    trigger_channel_ = modules.attr("_TRIGGER_CHANNEL");
    label_id_ = modules.attr("LABEL_ID");
    rf_shim_row_ = modules.attr("rf_shim_row");

    cursor_ = py::cast<int64_t>(owner.attr("cursor"));
    num_blocks_ = py::cast<int64_t>(owner.attr("num_blocks"));
    num_period_blocks_ = py::cast<int64_t>(owner.attr("num_period_blocks"));
    loop_size_ = py::cast<int64_t>(owner.attr("loop_size"));
    tag_ = py::cast<int64_t>(owner.attr("_tag"));
    ext_stride_ = py::cast<int>(owner.attr("_ext_stride"));

    slots_ = static_cast<const int32_t*>(hold(owner.attr("_slots"), 'i', "_slots").data);
    slots_ext_ =
        static_cast<const int32_t*>(hold(owner.attr("_slots_ext"), 'i', "_slots_ext").data);
    rf_ = static_cast<double*>(hold(owner.attr("_rf"), 'd', "_rf").data);
    trap_ = static_cast<double*>(hold(owner.attr("_trap"), 'd', "_trap").data);
    arb_ = static_cast<double*>(hold(owner.attr("_arb"), 'd', "_arb").data);
    adc_ = static_cast<double*>(hold(owner.attr("_adc"), 'd', "_adc").data);
    durations_ =
        static_cast<double*>(hold(owner.attr("_durations"), 'd', "_durations").data);

    on_ = checked_bytearray(keep(owner.attr("_on")), "_on");

    for (auto column : owner.attr("_ext_column")) {
      ext_column_.push_back(py::cast<int>(column));
    }
    for (auto width : owner.attr("_ext_widths")) {
      ext_widths_.push_back(py::cast<int>(width));
    }
    for (auto table : owner.attr("_ext_tables")) {
      py::object held = py::reinterpret_borrow<py::object>(table);
      ext_tables_.push_back(
          static_cast<double*>(hold(held, 'd', "_ext_tables entry").data));
    }
    for (auto switches : owner.attr("_ext_switch")) {
      py::object held = py::reinterpret_borrow<py::object>(switches);
      ext_switch_.push_back(checked_bytearray(keep(held), "_ext_switch entry"));
    }
  }

  int64_t cursor() const { return cursor_; }
  void set_cursor(int64_t value) { cursor_ = value; }

  // Takes the arguments as CPython hands them to a vectorcall, so that the raw
  // entry point below can pass them straight through. `add_block` is the same
  // method reached the ordinary way, and the two must not diverge.
  void write_block(PyObject* const* args, py::ssize_t nargs);

  // The same method reached through pybind11's ordinary dispatch. Kept as a
  // second way in, for anything holding the writer itself; the loop uses the
  // raw entry point, which is where the argument tuple this has to unpack is
  // never built in the first place.
  void add_block(py::args args) {
    const py::ssize_t given = PyTuple_GET_SIZE(args.ptr());
    std::vector<PyObject*> items(static_cast<size_t>(given));
    for (py::ssize_t n = 0; n < given; ++n) {
      items[static_cast<size_t>(n)] = PyTuple_GET_ITEM(args.ptr(), n);
    }
    write_block(items.data(), given);
  }

 private:
  py::object keep(py::object held) {
    alive_.push_back(held);
    return alive_.back();
  }

  Flat hold(py::object source, char kind, const char* what) {
    const Flat flat = flat_buffer(source, kind, what);
    alive_.push_back(std::move(source));
    return flat;
  }

  // Raise through the Python helper, which owns the wording. It always raises;
  // the throw after it is there so that no caller can fall through if it ever
  // stops doing so.
  [[noreturn]] void refuse(const py::object& helper, py::tuple arguments) {
    helper(*arguments);
    throw std::runtime_error("fastseq: a refusal helper returned instead of raising");
  }

  std::vector<py::object> alive_;

  py::object no_slot_, no_extension_, stale_copy_, wrong_shim_, adopt_;
  py::object trigger_type_, trigger_channel_, label_id_, rf_shim_row_;

  int64_t cursor_ = 0;
  int64_t num_blocks_ = 0;
  int64_t num_period_blocks_ = 0;
  int64_t loop_size_ = 0;
  int64_t tag_ = 0;
  int ext_stride_ = 0;

  const int32_t* slots_ = nullptr;
  const int32_t* slots_ext_ = nullptr;
  double* rf_ = nullptr;
  double* trap_ = nullptr;
  double* arb_ = nullptr;
  double* adc_ = nullptr;
  double* durations_ = nullptr;
  PyObject* on_ = nullptr;

  std::vector<int> ext_column_;
  std::vector<int> ext_widths_;
  std::vector<double*> ext_tables_;
  std::vector<PyObject*> ext_switch_;
};

void BlockWriter::write_block(PyObject* const* args, py::ssize_t count) {
  const int64_t i = cursor_;
  if (i >= num_blocks_) {
    throw py::index_error(
        "sequence holds " + std::to_string(num_blocks_) + " blocks (" +
        std::to_string(num_period_blocks_) + " per period x " +
        std::to_string(loop_size_) + " periods); add_block was called once too often");
  }

  const int64_t base = 5 * i;                       // _SLOT_COLUMNS
  char* on = PyByteArray_AS_STRING(on_);

  // One eight-bit field per extension type: how many of that type this block
  // has taken so far.
  uint64_t seen = 0;

  for (py::ssize_t n = 0; n < count; ++n) {
    PyObject* arg = args[n];

    // Subtracting the tag is the ownership check: an event this sequence
    // stamped lands in 1..15, anything else outside it. Adopting is what
    // happens to the outsiders that can be adopted -- delays and extensions,
    // which hold no waveform and name nothing -- and it happens once per
    // object, after which they take the fast path.
    int64_t code = 0;
    PyObject* stamp = try_getattr(arg, names().ps_code);
    if (stamp != nullptr) {
      if (PyLong_Check(stamp)) {
        code = PyLong_AsLongLong(stamp) - tag_;
      }
      Py_DECREF(stamp);
    }
    if (code < 1 || code > 15) {   // _PS_EXT_LAST
      code = py::cast<int64_t>(adopt_(py::reinterpret_borrow<py::object>(arg), i, tag_));
    }

    if (code < 4) {
      // Trapezoid: the commonest event, and the cheapest. Its code is already
      // the block column it belongs in.
      const int32_t at = slots_[base + code];
      if (at <= 0) {
        PyObject* channel = try_getattr(arg, names().channel);
        std::string what = "trapezoidal g";
        if (channel != nullptr) {
          what += py::cast<std::string>(py::reinterpret_steal<py::object>(channel));
        }
        refuse(no_slot_, py::make_tuple(i, at, what));
      }
      trap_[at - 1] = float_attr(arg, names().amplitude);
      on[base + code] = 1;

    } else if (code < 8) {
      if (code == 4) {                              // _PS_RF
        const int32_t slot = slots_[base];
        if (slot == 0) {
          refuse(no_slot_, py::make_tuple(i, slot, "RF pulse"));
        }
        double* rf = rf_ + (slot - 1);
        rf[0] = float_attr(arg, names().amplitude);
        rf[6] = float_attr(arg, names().freq_ppm);
        rf[7] = float_attr(arg, names().phase_ppm);
        rf[8] = float_attr(arg, names().freq_offset);
        rf[9] = float_attr(arg, names().phase_offset);
        on[base] = 1;
      } else {
        // Arbitrary waveform. The shape goes in unconditionally: every shot of
        // a multishot module has its own, and a slot holds whichever one it was
        // last handed.
        const int64_t column = code - 4;
        const int32_t slot = slots_[base + column];
        if (slot >= 0) {
          PyObject* channel = try_getattr(arg, names().channel);
          std::string what = "arbitrary g";
          if (channel != nullptr) {
            what += py::cast<std::string>(py::reinterpret_steal<py::object>(channel));
          }
          refuse(no_slot_, py::make_tuple(i, slot, what));
        }
        double* arb = arb_ + (-slot - 1);
        arb[0] = float_attr(arg, names().amplitude);
        arb[1] = float_attr(arg, names().first);
        arb[2] = float_attr(arg, names().last);
        PyObject* shape = try_getattr(arg, names().ps_shape);
        if (shape == nullptr) {
          refuse(stale_copy_, py::make_tuple(py::reinterpret_borrow<py::object>(arg), i));
        }
        arb[3] = as_double(shape);
        Py_DECREF(shape);
        on[base + column] = 1;
      }

    } else if (code == 8) {                         // _PS_ADC
      const int32_t slot = slots_[base + 4];
      if (slot == 0) {
        refuse(no_slot_, py::make_tuple(i, slot, "ADC"));
      }
      double* adc = adc_ + (slot - 1);
      adc[3] = float_attr(arg, names().freq_ppm);
      adc[4] = float_attr(arg, names().phase_ppm);
      adc[5] = float_attr(arg, names().freq_offset);
      adc[6] = float_attr(arg, names().phase_offset);
      // The phase-modulation shape, on the same terms as the arbitrary
      // gradient's. Zero is a real value here: it is what an ADC with no phase
      // modulation carries.
      PyObject* shape = try_getattr(arg, names().ps_shape);
      if (shape == nullptr) {
        refuse(stale_copy_, py::make_tuple(py::reinterpret_borrow<py::object>(arg), i));
      }
      adc[7] = as_double(shape);
      Py_DECREF(shape);
      on[base + 4] = 1;

    } else if (code == 9) {                         // _PS_DELAY
      durations_[i] = float_attr(arg, names().delay);

    } else {
      // An extension. Its code is its `EXTENSION_MAP` column, which
      // `_ext_column` turns into the column this sequence keeps a table for.
      const int64_t which = code - 10;              // _PS_EXT
      const int column =
          (which < static_cast<int64_t>(ext_column_.size())) ? ext_column_[which] : -1;
      if (column < 0) {
        refuse(no_extension_,
               py::make_tuple(i, py::reinterpret_borrow<py::object>(arg), 0));
      }

      const unsigned shift = static_cast<unsigned>(column) << 3;
      const uint64_t taken = (seen >> shift) & 255u;
      seen += uint64_t{1} << shift;

      const int64_t at = ext_stride_ * i + column + column;
      const int32_t allowed = slots_ext_[at + 1];
      if (static_cast<int64_t>(taken) >= allowed) {
        refuse(no_extension_,
               py::make_tuple(i, py::reinterpret_borrow<py::object>(arg), allowed));
      }
      const int64_t row = slots_ext_[at] + static_cast<int64_t>(taken);
      PyByteArray_AS_STRING(ext_switch_[column])[row] = 1;

      double* table = ext_tables_[column];
      if (table == nullptr) {
        continue;                                   // a type nothing varies about
      }
      const int width = ext_widths_[column];
      double* slot = table + row * width;

      if (code < 13) {                              // _PS_EXT_RF_SHIMS
        if (code > 10) {                            // _PS_EXT_TRIGGERS
          // A label: its value is what a scan loop moves, and its name goes in
          // beside it because which of a block's label slots this one landed in
          // is not fixed.
          slot[0] = float_attr(arg, names().value);
          PyObject* label = PyObject_GetAttr(arg, names().label.ptr());
          if (label == nullptr) {
            throw py::error_already_set();
          }
          PyObject* id = PyDict_GetItemWithError(label_id_.ptr(), label);
          Py_DECREF(label);
          if (id == nullptr) {
            if (!PyErr_Occurred()) {
              PyErr_SetObject(PyExc_KeyError, label);
            }
            throw py::error_already_set();
          }
          slot[1] = as_double(id);
        } else {
          PyObject* kind = PyObject_GetAttr(arg, names().type.ptr());
          if (kind == nullptr) {
            throw py::error_already_set();
          }
          PyObject* mapped = PyDict_GetItemWithError(trigger_type_.ptr(), kind);
          if (mapped == nullptr) {
            if (!PyErr_Occurred()) {
              PyErr_SetObject(PyExc_KeyError, kind);
            }
            Py_DECREF(kind);
            throw py::error_already_set();
          }
          Py_DECREF(kind);
          slot[0] = as_double(mapped);

          PyObject* channel = PyObject_GetAttr(arg, names().channel.ptr());
          if (channel == nullptr) {
            throw py::error_already_set();
          }
          PyObject* line = PyDict_GetItemWithError(trigger_channel_.ptr(), channel);
          if (line == nullptr) {
            if (!PyErr_Occurred()) {
              PyErr_SetObject(PyExc_KeyError, channel);
            }
            Py_DECREF(channel);
            throw py::error_already_set();
          }
          Py_DECREF(channel);
          slot[1] = as_double(line);

          slot[2] = float_attr(arg, names().delay);
          slot[3] = float_attr(arg, names().duration);
        }
      } else if (code == 14) {                      // _PS_EXT_ROTATIONS
        // Four numbers, scalar first. A `Rotation` is converted here if that is
        // what was handed in, but converting one costs more than the rest of
        // this method put together, so a loop that varies the rotation per
        // block should hand in the quaternion itself.
        PyObject* quaternion = PyObject_GetAttr(arg, names().rot_quaternion.ptr());
        if (quaternion == nullptr) {
          throw py::error_already_set();
        }
        py::object held = py::reinterpret_steal<py::object>(quaternion);
        PyObject* as_quat = try_getattr(held.ptr(), names().as_quat);
        if (as_quat != nullptr) {
          py::object call = py::reinterpret_steal<py::object>(as_quat);
          held = call(py::arg("canonical") = true, py::arg("scalar_first") = true);
        }
        for (int k = 0; k < 4; ++k) {
          PyObject* value = PySequence_GetItem(held.ptr(), k);
          if (value == nullptr) {
            throw py::error_already_set();
          }
          slot[k] = as_double(value);
          Py_DECREF(value);
        }
      } else {
        py::object values = rf_shim_row_(py::reinterpret_borrow<py::object>(arg));
        const py::ssize_t given = py::len(values);
        if (given != width) {
          refuse(wrong_shim_, py::make_tuple(i, given / 2, width / 2));
        }
        for (py::ssize_t k = 0; k < given; ++k) {
          PyObject* value = PySequence_GetItem(values.ptr(), k);
          if (value == nullptr) {
            throw py::error_already_set();
          }
          slot[k] = as_double(value);
          Py_DECREF(value);
        }
      }
    }
  }

  // Last, not first: a block that was refused half-way through is a block the
  // caller can fix and hand back, rather than one silently skipped.
  cursor_ = i + 1;
}

namespace {

// `add_block` as a plain CPython vectorcall, bypassing pybind11's dispatcher.
//
// That dispatcher is what makes a pybind11 binding pleasant everywhere else:
// it walks the overload chain, packs the arguments into a tuple, builds a call
// record and unpacks it again. It costs around two hundred nanoseconds, which
// is nothing at all until it is paid two million times -- on a large 3D scan it
// is most of a second, and more than the work of writing the blocks. A method
// taking `*args` and returning nothing needs none of it: CPython already has
// the arguments in an array, and `METH_FASTCALL` is how a C function says it
// will take them that way.
//
// The writer is reached through a capsule rather than through the pybind11
// object because a raw entry point gets whatever `self` it was created with,
// and a capsule can carry both the pointer and a reference that keeps it valid.
// The reference goes one way only -- the writer holds nothing back -- so the
// two of them do not form the kind of cycle that once made these sequences
// immortal (see the `BlockWriter` constructor).
struct FastBinding {
  BlockWriter* writer;
  py::object owner;
};

void release_binding(PyObject* capsule) {
  delete static_cast<FastBinding*>(PyCapsule_GetPointer(capsule, nullptr));
}

PyObject* fast_add_block(PyObject* self, PyObject* const* args, py::ssize_t nargs) {
  // An unnamed capsule, so this is a pointer read and two branches rather than
  // the string compare a named one would do on every block.
  auto* binding = static_cast<FastBinding*>(PyCapsule_GetPointer(self, nullptr));
  if (binding == nullptr) {
    return nullptr;
  }
  try {
    binding->writer->write_block(args, nargs);
  } catch (py::error_already_set& raised) {
    raised.restore();
    return nullptr;
  } catch (const py::builtin_exception& raised) {
    raised.set_error();
    return nullptr;
  } catch (const std::exception& raised) {
    PyErr_SetString(PyExc_RuntimeError, raised.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyMethodDef fast_add_block_def = {
    "add_block", reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(fast_add_block)),
    METH_FASTCALL, "Write one block of the unrolled scan, then advance the cursor."};

py::object bind_add_block(py::object self) {
  auto* binding = new FastBinding{py::cast<BlockWriter*>(self), std::move(self)};
  PyObject* capsule = PyCapsule_New(binding, nullptr, release_binding);
  if (capsule == nullptr) {
    delete binding;
    throw py::error_already_set();
  }
  py::object held = py::reinterpret_steal<py::object>(capsule);
  PyObject* bound = PyCFunction_New(&fast_add_block_def, capsule);
  if (bound == nullptr) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(bound);
}

}  // namespace

PYBIND11_MODULE(_fastseq_wrapper, m) {
  m.doc() = "Loop and deduplication accelerators for fastseq.";
  m.def("round_rows", &round_rows, py::arg("table"), py::arg("digits"),
        "Round a table to the precision that decides identity, column by column.");
  m.def("unique_rows", &unique_rows, py::arg("rows"),
        "The distinct rows in order of first appearance, and each row's index into them.");
  m.def("collapse_rows", &collapse_rows, py::arg("table"), py::arg("take"),
        py::arg("digits"), py::arg("extra") = py::none(),
        "Round and deduplicate the selected rows of a table without gathering them.");
  m.def("render_int_rows", &render_int_rows, py::arg("matrix"), py::arg("widths"),
        "A non-negative integer table as right-aligned space-separated lines.");
  m.def("tile_rows", &tile_rows, py::arg("period"), py::arg("loop_size"),
        "A table laid out end to end a number of times, as `np.tile` would.");
  m.def("block_columns", &block_columns, py::arg("slots"), py::arg("on"),
        py::arg("rf_width"), py::arg("adc_width"), py::arg("trap_width"),
        py::arg("arb_width"),
        "The block table's columns, and the index its gradient column points through.");

  py::class_<BlockWriter>(m, "BlockWriter")
      .def(py::init<py::object>(), py::arg("sequence"))
      .def_property("cursor", &BlockWriter::cursor, &BlockWriter::set_cursor)
      .def("add_block", &BlockWriter::add_block,
           "Write one block of the unrolled scan, then advance the cursor.")
      .def("bind", &bind_add_block,
           "`add_block` as a callable that CPython can reach without pybind11's dispatcher.");
}
