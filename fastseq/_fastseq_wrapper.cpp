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

  // Load factor one half, so probing stays short even when every row is unique.
  size_t capacity = 16;
  while (capacity < static_cast<size_t>(n) * 2) {
    capacity <<= 1;
  }
  const size_t mask = capacity - 1;

  std::vector<int64_t> slot(capacity, -1);
  std::vector<py::ssize_t> first;
  first.reserve(static_cast<size_t>(n) / 8 + 16);

  {
    py::gil_scoped_release unlocked;
    for (py::ssize_t r = 0; r < n; ++r) {
      const uint64_t* row = words + r * m;

      uint64_t h = 0xcbf29ce484222325ULL;
      for (py::ssize_t c = 0; c < m; ++c) {
        h ^= row[c];
        h *= 0x100000001b3ULL;
      }
      h ^= h >> 33;
      h *= 0xff51afd7ed558ccdULL;
      h ^= h >> 33;

      size_t at = static_cast<size_t>(h) & mask;
      for (;;) {
        const int64_t held = slot[at];
        if (held < 0) {
          slot[at] = static_cast<int64_t>(first.size());
          codes[r] = static_cast<int64_t>(first.size());
          first.push_back(r);
          break;
        }
        const uint64_t* candidate = words + first[static_cast<size_t>(held)] * m;
        if (std::memcmp(candidate, row, static_cast<size_t>(m) * 8) == 0) {
          codes[r] = held;
          break;
        }
        at = (at + 1) & mask;
      }
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

  Collapsed out;
  out.width = width;
  {
    py::gil_scoped_release unlocked;
    collapse_kernel(static_cast<const double*>(info.ptr), cols, picked, n, mode.data(),
                    scale.data(), power_table(), extra, width, out, code.mutable_data());
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

  // The widest each column gets, which is what bounds the buffer. A negative
  // anywhere means this cannot be rendered at a fixed width at all.
  std::vector<int> span(static_cast<size_t>(m));
  {
    py::gil_scoped_release unlocked;
    std::vector<int64_t> maxima(static_cast<size_t>(m), 0);
    for (py::ssize_t r = 0; r < n; ++r) {
      const int64_t* row = values + r * m;
      for (py::ssize_t c = 0; c < m; ++c) {
        if (row[c] < 0) {
          maxima[0] = -1;
          r = n;  // stop both loops
          break;
        }
        if (row[c] > maxima[static_cast<size_t>(c)]) {
          maxima[static_cast<size_t>(c)] = row[c];
        }
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
  out.resize(static_cast<size_t>(n) * line);  // an upper bound: shorter rows shrink it
  size_t used = 0;
  {
    py::gil_scoped_release unlocked;
    char* buffer = out.data();
    char digits[24];
    for (py::ssize_t r = 0; r < n; ++r) {
      const int64_t* row = values + r * m;
      for (py::ssize_t c = 0; c < m; ++c) {
        int64_t value = row[c];
        int count = 0;
        do {
          digits[count++] = static_cast<char>('0' + value % 10);
          value /= 10;
        } while (value != 0);
        for (int pad = widths[static_cast<size_t>(c)] - count; pad > 0; --pad) {
          buffer[used++] = ' ';
        }
        while (count > 0) {
          buffer[used++] = digits[--count];
        }
        buffer[used++] = (c + 1 == m) ? '\n' : ' ';
      }
    }
  }
  out.resize(used);
  return py::str(out);
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
  explicit BlockWriter(py::object owner) : owner_(std::move(owner)) {
    py::module_ modules = py::module_::import("fastseq.modules");
    no_slot_ = modules.attr("_no_slot");
    no_extension_ = modules.attr("_no_extension");
    stale_copy_ = modules.attr("_stale_copy");
    wrong_shim_ = modules.attr("_wrong_shim");
    adopt_ = modules.attr("_adopt");
    trigger_type_ = modules.attr("_TRIGGER_TYPE");
    trigger_channel_ = modules.attr("_TRIGGER_CHANNEL");
    label_id_ = modules.attr("LABEL_ID");
    rf_shim_row_ = modules.attr("rf_shim_row");

    cursor_ = py::cast<int64_t>(owner_.attr("cursor"));
    num_blocks_ = py::cast<int64_t>(owner_.attr("num_blocks"));
    num_period_blocks_ = py::cast<int64_t>(owner_.attr("num_period_blocks"));
    loop_size_ = py::cast<int64_t>(owner_.attr("loop_size"));
    tag_ = py::cast<int64_t>(owner_.attr("_tag"));
    ext_stride_ = py::cast<int>(owner_.attr("_ext_stride"));

    slots_ = static_cast<const int32_t*>(hold(owner_.attr("_slots"), 'i', "_slots").data);
    slots_ext_ =
        static_cast<const int32_t*>(hold(owner_.attr("_slots_ext"), 'i', "_slots_ext").data);
    rf_ = static_cast<double*>(hold(owner_.attr("_rf"), 'd', "_rf").data);
    trap_ = static_cast<double*>(hold(owner_.attr("_trap"), 'd', "_trap").data);
    arb_ = static_cast<double*>(hold(owner_.attr("_arb"), 'd', "_arb").data);
    adc_ = static_cast<double*>(hold(owner_.attr("_adc"), 'd', "_adc").data);
    durations_ =
        static_cast<double*>(hold(owner_.attr("_durations"), 'd', "_durations").data);

    on_ = checked_bytearray(keep(owner_.attr("_on")), "_on");

    // `cursor` stays an ordinary attribute of the sequence rather than becoming
    // a property that asks this object: everything else on the Python side
    // reads it, the pure-Python `_add_block_py` writes it, and having one place
    // it lives is worth the forty nanoseconds a write-back costs here.
    owner_dict_ = keep(py::reinterpret_borrow<py::object>(
        PyObject_GetAttrString(owner_.ptr(), "__dict__")));
    if (owner_dict_.ptr() == nullptr || !PyDict_Check(owner_dict_.ptr())) {
      throw std::invalid_argument("BlockWriter: the sequence has no instance dict");
    }
    cursor_key_ = Names::intern("cursor");

    for (auto column : owner_.attr("_ext_column")) {
      ext_column_.push_back(py::cast<int>(column));
    }
    for (auto width : owner_.attr("_ext_widths")) {
      ext_widths_.push_back(py::cast<int>(width));
    }
    for (auto table : owner_.attr("_ext_tables")) {
      py::object held = py::reinterpret_borrow<py::object>(table);
      ext_tables_.push_back(
          static_cast<double*>(hold(held, 'd', "_ext_tables entry").data));
    }
    for (auto switches : owner_.attr("_ext_switch")) {
      py::object held = py::reinterpret_borrow<py::object>(switches);
      ext_switch_.push_back(checked_bytearray(keep(held), "_ext_switch entry"));
    }
  }

  int64_t cursor() const { return cursor_; }
  void set_cursor(int64_t value) {
    cursor_ = value;
    publish_cursor();
  }

  void add_block(py::args args);

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

  void publish_cursor() {
    py::object value = py::reinterpret_steal<py::object>(PyLong_FromLongLong(cursor_));
    if (value.ptr() == nullptr ||
        PyDict_SetItem(owner_dict_.ptr(), cursor_key_.ptr(), value.ptr()) < 0) {
      throw py::error_already_set();
    }
  }

  // Raise through the Python helper, which owns the wording. It always raises;
  // the throw after it is there so that no caller can fall through if it ever
  // stops doing so.
  [[noreturn]] void refuse(const py::object& helper, py::tuple arguments) {
    helper(*arguments);
    throw std::runtime_error("fastseq: a refusal helper returned instead of raising");
  }

  py::object owner_;
  py::object owner_dict_;
  py::object cursor_key_;
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

void BlockWriter::add_block(py::args args) {
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

  const py::ssize_t count = PyTuple_GET_SIZE(args.ptr());
  for (py::ssize_t n = 0; n < count; ++n) {
    PyObject* arg = PyTuple_GET_ITEM(args.ptr(), n);

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
  publish_cursor();
}

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

  py::class_<BlockWriter>(m, "BlockWriter")
      .def(py::init<py::object>(), py::arg("sequence"))
      .def_property("cursor", &BlockWriter::cursor, &BlockWriter::set_cursor)
      .def("add_block", &BlockWriter::add_block,
           "Write one block of the unrolled scan, then advance the cursor.");
}
