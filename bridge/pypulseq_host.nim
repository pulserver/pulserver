## pypulseq_host — pre-compiled generic host for Python-based Pulseq plugins.
##
## Compiled once per platform. Loads a user-supplied ``.py`` script at runtime
## via ``--script`` flag. The Python script provides the three sequence
## callbacks (``get_default_protocol``, ``validate_protocol``,
## ``make_sequence``); this host wraps them and dispatches to either
## nimpulseqgui's interactive GUI or the bridge's own headless modes
## (``--validate-only``, ``--persistent``, ``--list-protocol``).
##
## **Key design**: nimpulseqgui is used as-is (stock dependency).
## Rich ``ValidationResult`` flows through headless paths only;
## for the GUI path the bridge wraps it down to ``bool``.

import bridge_common
import nimpy
import nimpy/py_lib as nimpy_lib
import std/[dynlib, os, strutils, parseopt, times]
import posix

proc isPyNone*(o: PyObject): bool =
  ## Returns true if the Python object is ``None``.
  if o.isNil: return true
  let none = pyBuiltinsModule().getAttr("None")
  return o == none

const pythonHome {.strdefine.} = "python"
  ## Relative path to a bundled Python venv (dir name), checked relative to exe dir
  ## and its parent.  Used for developer/test layouts.

const pythonVenvPath {.strdefine.} = "/usr/g/bin/recon/research/EnvPulserver"
  ## Absolute path to the canonical scanner venv for pypulseq_host.
  ## Override at compile time with -d:pythonVenvPath=<path>.

type BundledPythonKind* = enum
  bpkNone, bpkVenv, bpkStandalone

type BundledPythonEnv* = object
  kind*: BundledPythonKind
  home*: string  ## Root of the python dir (venv root or standalone prefix)

proc resolveBundledPythonHome*(exeDir: string; canonicalVenvPath: string = pythonVenvPath): BundledPythonEnv =
  ## Resolves the Python environment for this process.
  ## Search order:
  ##   1. Canonical absolute venv path (pythonVenvPath compile-time define)
  ##   2. <exe_dir>/<pythonHome>  (developer/test layout)
  ##   3. <exe_dir>/../<pythonHome>  (installer layout: exe in bin/, venv at sibling)
  let candidates = @[
    canonicalVenvPath,
    exeDir / pythonHome,
    parentDir(exeDir) / pythonHome,
  ]
  for candidate in candidates:
    if dirExists(candidate):
      if fileExists(candidate / "pyvenv.cfg"):
        return BundledPythonEnv(kind: bpkVenv, home: candidate)
      elif dirExists(candidate / "lib") and dirExists(candidate / "bin"):
        return BundledPythonEnv(kind: bpkStandalone, home: candidate)
  return BundledPythonEnv(kind: bpkNone, home: "")

proc findLibPythonInVenv*(venvHome: string): tuple[libPython, baseLibDir: string] =
  ## Locates the ``libpython3.X.so`` (or ``.so.1.0``) that the venv's
  ## Python installation provides, plus the base lib directory.
  ## Returns empty strings if not found.
  ##
  ## Strategy: parse ``pyvenv.cfg``'s ``home`` and ``version`` keys.
  ## Try the venv's exact major.minor first so we don't accidentally pick a
  ## higher-versioned libpython from a multi-python installation (which would
  ## mismatch the site-packages under lib/pythonX.Y/).  Fall back to the
  ## full version list only if the exact match is absent.
  let cfgPath = venvHome / "pyvenv.cfg"
  if not fileExists(cfgPath):
    return ("", "")
  var pythonBinDir = ""
  var venvMajMin = ""
  for line in lines(cfgPath):
    let stripped = line.strip()
    if stripped.startsWith("home ") or stripped.startsWith("home="):
      let parts = stripped.split('=', maxsplit = 1)
      if parts.len == 2 and parts[0].strip() == "home":
        pythonBinDir = parts[1].strip()
    elif stripped.startsWith("version ") or stripped.startsWith("version="):
      let parts = stripped.split('=', maxsplit = 1)
      if parts.len == 2 and parts[0].strip() == "version":
        let dots = parts[1].strip().split('.')
        if dots.len >= 2:
          venvMajMin = dots[0] & "." & dots[1]
  if pythonBinDir.len == 0:
    return ("", "")
  # The lib dir is <prefix>/lib on most distros, but <prefix>/lib64 on
  # SUSE/RPM-based systems (x86_64).  Check lib64 first so the scanner's
  # bundled Python (extracted from SUSE RPMs into usr/lib64/) is found.
  let prefix = parentDir(pythonBinDir)
  var libDir = ""
  for candidate in [prefix / "lib64", prefix / "lib"]:
    if dirExists(candidate):
      libDir = candidate
      break
  if libDir.len == 0:
    return ("", "")
  # Pass 1: exact venv version (avoids grabbing a mismatched higher version).
  if venvMajMin.len > 0:
    for suffix in [".so.1.0", ".so.1", ".so"]:
      let candidate = libDir / "libpython" & venvMajMin & suffix
      if fileExists(candidate):
        return (candidate, libDir)
  # Pass 2: fallback — highest available (handles incomplete installations).
  for v in ["3.13", "3.12", "3.11", "3.10", "3.9", "3.8"]:
    if v == venvMajMin: continue  # already checked in pass 1
    for suffix in [".so.1.0", ".so.1", ".so"]:
      let candidate = libDir / "libpython" & v & suffix
      if fileExists(candidate):
        return (candidate, libDir)
  return ("", libDir)

# ── Marshalling: Nim → Python ──────────────────────────────────────────────

proc nimOptsToPyOpts*(opts: Opts): PyObject =
  ## Constructs a ``pypulseq.Opts`` Python object from Nim ``Opts``.
  ## Both sides store maxGrad/maxSlew in Hz/m and Hz/m/s respectively.
  let pp = pyImport("pypulseq")
  result = pp.Opts(
    max_grad = opts.maxGrad,
    max_slew = opts.maxSlew,
    grad_raster_time = opts.gradRasterTime,
    rf_dead_time = opts.rfDeadTime,
    rf_ringdown_time = opts.rfRingdownTime,
    adc_dead_time = opts.adcDeadTime,
    adc_raster_time = opts.adcRasterTime,
    rf_raster_time = opts.rfRasterTime,
    block_duration_raster = opts.blockDurationRaster,
    B0 = opts.B0,
    gamma = opts.gamma)

proc parseFloatOption(value: string; optionName: string): float64 =
  try:
    result = parseFloat(value)
  except ValueError:
    raise newException(ValueError, "Invalid value for --" & optionName & ": " & value)

proc parseIntOption(value: string; optionName: string): int =
  try:
    result = parseInt(value)
  except ValueError:
    raise newException(ValueError, "Invalid value for --" & optionName & ": " & value)

proc splitLongOption(argument: string): tuple[name: string, value: string, hasValue: bool] =
  var option = argument
  if option.startsWith("--"):
    option = option[2..^1]
  elif option.startsWith("-"):
    option = option[1..^1]
  else:
    return ("", "", false)

  let parts = option.split("=", maxsplit = 1)
  if parts.len == 2:
    return (parts[0], parts[1], true)
  return (option, "", false)

proc optsFromForwardArgs*(args: seq[string]): Opts =
  ## Builds scanner options for bridge-owned headless modes from the same
  ## long options that nimpulseqgui consumes in GUI mode.
  var maxGrad = -1.0
  var maxSlew = -1.0
  var riseTime = -1.0
  var rfDeadTime = -1.0
  var rfRingdownTime = -1.0
  var adcDeadTime = -1.0
  var adcRasterTime = -1.0
  var rfRasterTime = -1.0
  var gradRasterTime = -1.0
  var blockDurationRaster = -1.0
  var gamma = -1.0
  var b0 = -1.0
  var adcSamplesLimit = -1
  var adcSamplesDivisor = -1
  var gradUnit = "Hz/m"
  var slewUnit = "Hz/m/s"

  var index = 0
  while index < args.len:
    let argument = args[index]
    if not argument.startsWith("-"):
      inc index
      continue

    var (optionName, optionValue, hasValue) = splitLongOption(argument)
    if optionName.len == 0:
      inc index
      continue
    if not hasValue and index + 1 < args.len and not args[index + 1].startsWith("-"):
      optionValue = args[index + 1]
      hasValue = true
      inc index

    case optionName
    of "maxGrad":
      if hasValue: maxGrad = parseFloatOption(optionValue, optionName)
    of "maxSlew":
      if hasValue: maxSlew = parseFloatOption(optionValue, optionName)
    of "riseTime":
      if hasValue: riseTime = parseFloatOption(optionValue, optionName)
    of "rfDeadTime":
      if hasValue: rfDeadTime = parseFloatOption(optionValue, optionName)
    of "rfRingdownTime":
      if hasValue: rfRingdownTime = parseFloatOption(optionValue, optionName)
    of "adcDeadTime":
      if hasValue: adcDeadTime = parseFloatOption(optionValue, optionName)
    of "adcRasterTime":
      if hasValue: adcRasterTime = parseFloatOption(optionValue, optionName)
    of "rfRasterTime":
      if hasValue: rfRasterTime = parseFloatOption(optionValue, optionName)
    of "gradRasterTime":
      if hasValue: gradRasterTime = parseFloatOption(optionValue, optionName)
    of "blockDurationRaster":
      if hasValue: blockDurationRaster = parseFloatOption(optionValue, optionName)
    of "adcSamplesLimit":
      if hasValue: adcSamplesLimit = parseIntOption(optionValue, optionName)
    of "adcSamplesDivisor":
      if hasValue: adcSamplesDivisor = parseIntOption(optionValue, optionName)
    of "gamma":
      if hasValue: gamma = parseFloatOption(optionValue, optionName)
    of "B0":
      if hasValue: b0 = parseFloatOption(optionValue, optionName)
    of "gradUnit":
      if hasValue: gradUnit = optionValue
    of "slewUnit":
      if hasValue: slewUnit = optionValue
    else:
      discard
    inc index

  return newOpts(
    maxGrad = maxGrad,
    gradUnit = gradUnit,
    maxSlew = maxSlew,
    slewUnit = slewUnit,
    rfRingdownTime = rfRingdownTime,
    rfDeadTime = rfDeadTime,
    adcDeadTime = adcDeadTime,
    adcRasterTime = adcRasterTime,
    rfRasterTime = rfRasterTime,
    gradRasterTime = gradRasterTime,
    blockDurationRaster = blockDurationRaster,
    riseTime = riseTime,
    gamma = gamma,
    B0 = b0,
    adcSamplesLimit = adcSamplesLimit,
    adcSamplesDivisor = adcSamplesDivisor)

proc protToPyDict*(prot: MRProtocolRef): PyObject =
  ## Serializes ``MRProtocolRef`` → Python dict of property dicts.
  let builtins = pyBuiltinsModule()
  result = builtins.callMethod("dict")
  for key, prop in prot:
    let d = builtins.callMethod("dict")
    case prop.pType
    of ptInt:
      d["type"]  = "int";   d["value"] = prop.intVal
      d["min"]   = prop.intMin;  d["max"] = prop.intMax; d["incr"] = prop.intIncr
    of ptFloat:
      d["type"]  = "float"; d["value"] = prop.floatVal
      d["min"]   = prop.floatMin; d["max"] = prop.floatMax; d["incr"] = prop.floatIncr
    of ptBool:
      d["type"]  = "bool";  d["value"] = prop.boolVal
    of ptStringList:
      d["type"]  = "stringlist"; d["value"] = prop.stringVal
      d["options"] = prop.stringList; d["index"] = prop.stringList.find(prop.stringVal)
    of ptDescription:
      d["type"]  = "description"; d["text"] = prop.description
    d["unit"]      = prop.unit
    d["validate"]  = (if prop.validateStrategy == pvDoSearch: "search" else: "none")
    result[key]    = d

proc makeProtocolSchemaPreamble*(prot: MRProtocolRef): string =
  ## Serializes protocol values with schema metadata for the GE bridge.
  ##
  ## NOTE: ``MRProtocolRef`` cannot represent the ``mode``/``options``
  ## metadata used by the GE interpreter to render dropdown widgets. The
  ## headless paths therefore prefer ``makeProtocolSchemaPreambleFromPy``,
  ## which serializes directly from the original Python dict and preserves
  ## that metadata. This MRProtocolRef-based emitter is retained as a
  ## fallback for callers that only have a Nim protocol in hand.
  var lines: seq[string] = @[protocolPreambleStart]
  for key, prop in prot:
    case prop.pType
    of ptInt:
      lines.add(
        key & ": int|typein|" & $prop.intVal & "|" & $prop.intMin & "|" &
        $prop.intMax & "|" & $prop.intIncr & "|" & prop.unit)
    of ptFloat:
      lines.add(
        key & ": float|typein|" & $prop.floatVal & "|" & $prop.floatMin & "|" &
        $prop.floatMax & "|" & $prop.floatIncr & "|" & prop.unit)
    of ptBool:
      lines.add(key & ": bool|" & (if prop.boolVal: "true" else: "false"))
    of ptStringList:
      var fields = @[key & ": stringlist", $prop.stringList.find(prop.stringVal)]
      for option in prop.stringList:
        fields.add(option)
      lines.add(fields.join("|"))
    of ptDescription:
      lines.add(key & ": description|" & prop.description.replace("\n", "\\n"))
  lines.add(protocolPreambleEnd)
  result = lines.join("\n")

proc pyHasKey(d: PyObject, key: string): bool =
  ## Returns true if Python dict *d* contains *key*.
  let builtins = pyBuiltinsModule()
  return builtins.callMethod("bool", d.callMethod("__contains__", key)).to(bool)

proc pyDictGet(d: PyObject, key: string, default: string): string {.gcsafe.}
  ## Forward declaration; implementation below in the Python→Nim section.

proc pyModeString(d: PyObject): string =
  ## Returns the wire-format mode token ("typein"/"dropdown"/"off") for a
  ## numeric protocol entry. Python ``InputMode`` is a ``StrEnum`` whose
  ## values match the wire-format tokens exactly; use ``str()`` on the value
  ## to obtain the canonical token regardless of whether ``asdict`` left an
  ## enum instance or a plain str.
  if not pyHasKey(d, "mode"):
    return "typein"
  let raw = d["mode"]
  let builtins = pyBuiltinsModule()
  var s = ""
  try:
    if builtins.callMethod("hasattr", raw, "value").to(bool):
      s = builtins.callMethod("str", raw.getAttr("value")).to(string)
    else:
      s = builtins.callMethod("str", raw).to(string)
  except CatchableError:
    s = builtins.callMethod("str", raw).to(string)
  s = s.strip().toLowerAscii()
  if s.startsWith("inputmode."):
    s = s.split(".", maxsplit = 1)[1]
  if s in ["off", "typein", "dropdown"]:
    return s
  return "typein"

proc makeProtocolSchemaPreambleFromPy*(pyDict: PyObject): string =
  ## Serializes a Python protocol dict (as produced by ``protocol_to_dict``)
  ## with full schema metadata for the GE bridge, including dropdown
  ## ``mode``/``options`` for int and float entries.
  var lines: seq[string] = @[protocolPreambleStart]
  for key in pyDict:
    let k = key.to(string)
    let d = pyDict[key]
    let ptype = d["type"].to(string)
    case ptype
    of "int":
      let mode = pyModeString(d)
      var line = k & ": int|" & mode & "|" & $d["value"].to(int) & "|" &
        $d["min"].to(int) & "|" & $d["max"].to(int) & "|" &
        $d["incr"].to(int) & "|" & pyDictGet(d, "unit", "")
      if mode == "dropdown" and pyHasKey(d, "options"):
        for opt in d["options"].to(seq[int]):
          line.add("|" & $opt)
      lines.add(line)
    of "float":
      let mode = pyModeString(d)
      var line = k & ": float|" & mode & "|" & $d["value"].to(float) & "|" &
        $d["min"].to(float) & "|" & $d["max"].to(float) & "|" &
        $d["incr"].to(float) & "|" & pyDictGet(d, "unit", "")
      if mode == "dropdown" and pyHasKey(d, "options"):
        for opt in d["options"].to(seq[float]):
          line.add("|" & $opt)
      lines.add(line)
    of "bool":
      lines.add(k & ": bool|" & (if d["value"].to(bool): "true" else: "false"))
    of "stringlist":
      let options = d["options"].to(seq[string])
      let selected = d["value"].to(string)
      var fields = @[k & ": stringlist", $options.find(selected)]
      for option in options:
        fields.add(option)
      lines.add(fields.join("|"))
    of "description":
      let text = pyDictGet(d, "text", "")
      lines.add(k & ": description|" & text.replace("\n", "\\n"))
    else:
      discard
  lines.add(protocolPreambleEnd)
  result = lines.join("\n")

# ── Marshalling: Python → Nim ──────────────────────────────────────────────

proc pyDictGet(d: PyObject, key: string, default: string): string =
  ## Safely get a string from a Python dict, returning *default* if missing.
  return d.callMethod("get", key, default).to(string)

proc pyDictToProt*(pyDict: PyObject): MRProtocolRef =
  ## Deserializes Python dict → ``MRProtocolRef``.
  result = newProtocol()
  for key in pyDict:
    let k = key.to(string)
    let d = pyDict[key]
    let ptype = d["type"].to(string)
    let validate = if pyDictGet(d, "validate", "none") == "search": pvDoSearch else: pvNoSearch
    let unit = pyDictGet(d, "unit", "")
    case ptype
    of "int":
      result[k] = newIntProperty(d["value"].to(int), d["min"].to(int),
                                  d["max"].to(int), d["incr"].to(int),
                                  validate=validate, unit=unit)
    of "float":
      result[k] = newFloatProperty(d["value"].to(float), d["min"].to(float),
                                    d["max"].to(float), d["incr"].to(float),
                                    validate=validate, unit=unit)
    of "bool":
      result[k] = newBoolProperty(d["value"].to(bool), validate=validate)
    of "stringlist":
      let options = d["options"].to(seq[string])
      result[k] = newStringListProperty(d["value"].to(string), options, validate=validate)
    of "description":
      result[k] = newDescriptionProperty(d["text"].to(string))
    else:
      echo "WARNING: unknown property type '", ptype, "' for key '", k, "'"

# ── Python module loader ──────────────────────────────────────────────────

type PyPlugin* = object
  module*: PyObject

proc loadPyPlugin*(scriptPath: string): PyPlugin =
  ## Loads a Python plugin module from *scriptPath* via ``importlib``.
  let importlib = pyImport("importlib.util")
  let spec = importlib.callMethod("spec_from_file_location", "plugin", scriptPath)
  let module = importlib.callMethod("module_from_spec", spec)
  discard spec.loader.callMethod("exec_module", module)
  result.module = module

proc addPythonPath*(path: string) =
  ## Prepends *path* to ``sys.path`` to support local plugin imports.
  let sys = pyImport("sys")
  discard sys.path.callMethod("insert", 0, path)

# ── Callback wrappers ─────────────────────────────────────────────────────

proc wrapGetDefaultProtocol*(plugin: PyPlugin): ProcGetDefaultProtocol =
  ## Wraps ``plugin.get_default_protocol(pyOpts) → dict`` as ``ProcGetDefaultProtocol``.
  return proc(opts: Opts): MRProtocolRef =
    let pyResult = plugin.module.get_default_protocol(nimOptsToPyOpts(opts))
    return pyDictToProt(pyResult)

proc getDefaultProtocolPy*(plugin: PyPlugin, opts: Opts): PyObject =
  ## Returns the raw Python dict from ``plugin.get_default_protocol`` so that
  ## headless callers can preserve ``mode``/``options`` metadata when
  ## emitting the GE bridge preamble.
  return plugin.module.get_default_protocol(nimOptsToPyOpts(opts))

proc wrapValidateRich*(plugin: PyPlugin): ProcValidateRich =
  ## Wraps ``plugin.validate_protocol(pyOpts, prot_dict) → dict`` as ``ProcValidateRich``.
  ## Returns the full ``ValidationResult{valid, duration, info}``.
  return proc(opts: Opts, prot: MRProtocolRef): ValidationResult =
    let pyResult = plugin.module.validate_protocol(
      nimOptsToPyOpts(opts), protToPyDict(prot))
    result.valid = pyResult["valid"].to(bool)
    let pyDur = pyResult["duration"]
    result.duration = if pyDur.isPyNone(): -1.0 else: pyDur.to(float)
    let pyInfo = pyResult["info"]
    result.info = if pyInfo.isPyNone(): "" else: pyInfo.to(string)

proc wrapValidateProtocol*(plugin: PyPlugin): ProcValidateProtocol =
  ## Wraps the Python validator as nimpulseqgui's ``ProcValidateProtocol`` (bool).
  ## Used when delegating to the GUI.
  let richValidator = wrapValidateRich(plugin)
  return proc(opts: Opts, prot: MRProtocolRef): bool =
    return richValidator(opts, prot).valid

proc callMakeSequenceFile*(plugin: PyPlugin, opts: Opts, prot: MRProtocolRef, outPath: string) =
  ## Calls Python ``make_sequence(opts, protocol, output_path)``.
  ## The plugin writes the ``.seq`` file to *outPath* directly.
  discard plugin.module.make_sequence(
    nimOptsToPyOpts(opts), protToPyDict(prot), outPath)

proc wrapMakeSequence*(plugin: PyPlugin): ProcMakeSequence =
  ## Wraps ``plugin.make_sequence`` as nimpulseqgui's ``ProcMakeSequence``.
  ##
  ## nimpulseqgui expects ``ProcMakeSequence`` to return a ``Sequence`` object
  ## (nimpulseq in-memory representation) so it can call ``writeSeq`` and
  ## prepend the protocol preamble. Since nimpulseq is write-only (no
  ## ``readSeq``), this wrapper cannot produce the required return type.
  ## Blocked on nimpulseq adding ``readSeq``.
  ##
  ## Headless modes bypass this entirely via ``callMakeSequenceFile``.
  return proc(opts: Opts, prot: MRProtocolRef): Sequence =
    raise newException(Defect,
      "GUI-path make_sequence blocked on nimpulseq readSeq. " &
      "Use headless modes (--persistent, --validate-only, --no-gui) for now.")

# ── CLI ────────────────────────────────────────────────────────────────────

proc printBridgeHelp() =
  echo "pypulseq_host — Python plugin host for nimpulseqgui"
  echo ""
  echo "Usage: pypulseq_host --script <plugin.py> [options]"
  echo ""
  echo "Bridge-specific options:"
  echo "  --script=<path>          Path to the Python plugin .py file (required)"
  echo "  -p, --protocol=<text>    Inline protocol preamble string"
  echo "  --validate-only          Validate protocol, print JSON result, and exit"
  echo "  --list-protocol          Print default protocol preamble and exit"
  echo "  --persistent             Persistent stdin/stdout command loop (for GE popen)"
  echo ""
  echo "All nimpulseqgui flags (--output, --input, --manufacturer, etc.) are also accepted."
  echo "When none of the bridge-specific headless flags are given, falls through to"
  echo "nimpulseqgui's makeSequenceExe (GUI or --no-gui)."

proc main() =
  # Init bundled Python environment.
  # Venv: do NOT set PYTHONHOME (breaks venv); set VIRTUAL_ENV + PYTHONPATH instead.
  # Standalone: set PYTHONHOME to the prefix (original behaviour).
  let bundledPy = resolveBundledPythonHome(getAppDir())
  case bundledPy.kind
  of bpkVenv:
    # Activate the venv for this process: VIRTUAL_ENV, updated PATH, PYTHONPATH
    # to site-packages so nimpy's embedded interpreter finds packages correctly.
    putEnv("VIRTUAL_ENV", bundledPy.home)
    let venvBin = bundledPy.home / "bin"
    let existingPath = getEnv("PATH")
    putEnv("PATH", venvBin & ":" & existingPath)
    # Locate site-packages and expose it via PYTHONPATH.
    let libDir = bundledPy.home / "lib"
    var sitePkgs = ""
    for kind, p in walkDir(libDir):
      if kind == pcDir and p.lastPathPart.startsWith("python"):
        let sp = p / "site-packages"
        if dirExists(sp):
          sitePkgs = sp
          break
    if sitePkgs.len > 0:
      let existingPythonPath = getEnv("PYTHONPATH")
      if existingPythonPath.len > 0:
        putEnv("PYTHONPATH", sitePkgs & ":" & existingPythonPath)
      else:
        putEnv("PYTHONPATH", sitePkgs)
    # Tell nimpy to load the libpython from the venv's base Python installation
    # instead of whatever the system linker finds first.  This is critical when
    # the venv's Python version differs from the system Python.
    # Also preload shared libraries from the base installation's lib dir with
    # RTLD_GLOBAL so that Python extension modules (e.g. pyexpat) find their
    # native dependencies (e.g. libexpat) from the same installation.
    # putEnv("LD_LIBRARY_PATH") does NOT work because glibc caches the value
    # at process startup; explicit preloading via dlopen is required.
    let (venvLibPython, baseLibDir) = findLibPythonInVenv(bundledPy.home)
    if baseLibDir.len > 0:
      # Preload only the shared libraries that Python's C extension modules
      # commonly link against.  We avoid loading everything in lib/ because
      # compiler runtimes (libasan, libhwasan, libtsan, …) can break the
      # process.  The list below covers the standard-library extensions that
      # ship with CPython (pyexpat, _ssl, zlib, _bz2, _lzma, _sqlite3,
      # _ctypes, readline, _curses, _hashlib, _uuid, _dbm).
      const preloadPrefixes = [
        "libexpat.so", "libz.so", "libssl.so", "libcrypto.so",
        "libffi.so", "libbz2.so", "liblzma.so", "libsqlite3.so",
        "libreadline.so", "libncurses.so", "libncursesw.so",
        "libuuid.so", "libgdbm.so", "libmpdec.so", "libmpdec++.so",
      ]
      for kind, p in walkDir(baseLibDir):
        if kind != pcFile:
          continue
        let name = p.lastPathPart
        for prefix in preloadPrefixes:
          if name.startsWith(prefix):
            discard loadLib(p, globalSymbols = true)
            break
    if venvLibPython.len > 0:
      nimpy_lib.pyInitLibPath(venvLibPython)
  of bpkStandalone:
    putEnv("PYTHONHOME", bundledPy.home)
  of bpkNone:
    discard  # Use system Python

  # ── Parse bridge-specific flags (consumed here, rest forwarded) ──
  var scriptPath = ""
  var inputProtocolString = ""
  var validateOnly = false
  var listProtocol = false
  var persistentMode = false
  var idleTimeoutSecs = 0  ## 0 = disabled
  var forwardArgs: seq[string] = @[]  # args to forward to makeSequenceExe
  var expectScriptPath = false
  var expectProtocolString = false
  var expectIdleTimeout = false

  for kind, key, val in getopt():
    case kind
    of cmdArgument:
      if expectScriptPath:
        scriptPath = key
        expectScriptPath = false
      elif expectProtocolString:
        inputProtocolString = key
        expectProtocolString = false
      elif expectIdleTimeout:
        idleTimeoutSecs = parseInt(key)
        expectIdleTimeout = false
      else:
        forwardArgs.add(key)
    of cmdLongOption, cmdShortOption:
      case key
      of "script":
        if val.len > 0: scriptPath = val
        else: expectScriptPath = true
      of "protocol", "p":
        if val.len > 0: inputProtocolString = val
        else: expectProtocolString = true
      of "validate-only":
        validateOnly = true
      of "list-protocol":
        listProtocol = true
      of "persistent":
        persistentMode = true
      of "idle-timeout":
        if val.len > 0: idleTimeoutSecs = parseInt(val)
        else: expectIdleTimeout = true
      of "h", "help":
        printBridgeHelp()
        quit(0)
      else:
        # Forward to makeSequenceExe
        if val.len > 0:
          forwardArgs.add("--" & key & "=" & val)
        else:
          forwardArgs.add("--" & key)
    of cmdEnd:
      discard

  if scriptPath == "":
    echo "Error: --script is required."
    printBridgeHelp()
    quit(1)

  # Apply env var fallback for idle timeout (used when spawned via bridge API
  # which does not support extra CLI flags beyond scanner opts).
  if idleTimeoutSecs == 0:
    let envTimeout = getEnv("PYPULSEQ_HOST_IDLE_TIMEOUT", "")
    if envTimeout.len > 0:
      try: idleTimeoutSecs = parseInt(envTimeout)
      except ValueError: discard

  let pluginScriptPath = scriptPath

  if not fileExists(pluginScriptPath):
    echo "Error: plugin file not found: " & pluginScriptPath
    quit(1)

  # Allow relative imports next to the plugin script.
  addPythonPath(parentDir(pluginScriptPath))

  let plugin = loadPyPlugin(pluginScriptPath)
  let getDefault = wrapGetDefaultProtocol(plugin)
  let validateBool = wrapValidateProtocol(plugin)
  let validateRich = wrapValidateRich(plugin)
  let makeSeq = wrapMakeSequence(plugin)  # GUI path only (blocked on readSeq)

  # ── Headless paths: bridge owns the CLI, gets full ValidationResult ──

  let headless = validateOnly or listProtocol or persistentMode or
                 inputProtocolString.len > 0

  if headless:
    let opts = optsFromForwardArgs(forwardArgs)
    let defaultProtPy = getDefaultProtocolPy(plugin, opts)
    let defaultProt = pyDictToProt(defaultProtPy)
    if not validateBool(opts, defaultProt):
      echo "FATAL ERROR: default protocol is not valid!"
      quit(1)

    # Apply inline protocol if given
    if inputProtocolString.len > 0:
      var prot = defaultProt.copy
      let warnings = readProtocolFromString(inputProtocolString, opts, prot, validateBool)
      for w in warnings: echo w

      if validateOnly:
        let vr = validateRich(opts, prot)
        echo formatValidationJson(vr)
        quit(if vr.valid: 0 else: 1)

    elif validateOnly:
      let vr = validateRich(opts, defaultProt)
      echo formatValidationJson(vr)
      quit(if vr.valid: 0 else: 1)

    if listProtocol:
      echo "PROTOCOL"
      echo makeProtocolSchemaPreambleFromPy(defaultProtPy)
      quit(0)

    if persistentMode:
      # Stateless stdin/stdout command loop.
      # Each command starts from defaultProt (the plugin's fixed schema).
      # No state is carried between commands — persistence is purely about
      # keeping the Python interpreter warm.
      #
      # Idle timeout: if idleTimeoutSecs > 0, exit cleanly after that many
      # seconds of inactivity (no command received).
      var lastActivity = epochTime()
      let stdinFd = cint(0)  # STDIN_FILENO
      while true:
        var cmd: string
        if idleTimeoutSecs > 0:
          # Poll stdin with a 5-second tick so we can check idle time.
          var readSet: TFdSet
          FD_ZERO(readSet)
          FD_SET(stdinFd, readSet)
          var tv: Timeval
          tv.tv_sec = posix.Time(5)
          tv.tv_usec = 0
          let ready = posix.select(stdinFd + 1, addr readSet, nil, nil, addr tv)
          if ready == 0:
            # Timeout tick — check idle duration
            let idle = epochTime() - lastActivity
            if idle >= float(idleTimeoutSecs):
              break  # exit cleanly
            continue
          elif ready < 0:
            break  # select error
          cmd = stdin.readLine().strip()
          lastActivity = epochTime()
        else:
          cmd = stdin.readLine().strip()
        if cmd == "QUIT" or cmd == "":
          break
        elif cmd == "LIST_PROTOCOL":
          echo "PROTOCOL"
          echo makeProtocolSchemaPreambleFromPy(defaultProtPy)
          flushFile(stdout)
        elif cmd == "VALIDATE":
          let preambleStr = readPreambleFromStdin()
          var localProt = defaultProt.copy
          discard readProtocolFromString(preambleStr, opts, localProt, validateBool)
          let vr = validateRich(opts, localProt)
          echo formatValidationPlain(vr)
          flushFile(stdout)
        elif cmd.startsWith("GENERATE "):
          let outPath = cmd[9..^1].strip()
          let preambleStr = readPreambleFromStdin()
          var localProt = defaultProt.copy
          discard readProtocolFromString(preambleStr, opts, localProt, validateBool)
          try:
            callMakeSequenceFile(plugin, opts, localProt, outPath)
            echo "GENERATED " & outPath
          except Exception as e:
            echo "ERROR " & e.msg
          flushFile(stdout)
        else:
          echo "ERROR unknown command: " & cmd
          flushFile(stdout)
      quit(0)

  # ── GUI path: delegate to stock makeSequenceExe (bool validator) ──
  # Re-inject forwarded args so makeSequenceExe can parse them.
  # nimpy + nimpulseqgui both use getopt() which reads commandLine().
  makeSequenceExe(getDefault, validateBool, makeSeq,
                  "PyPulseq: " & pluginScriptPath.extractFilename())

when isMainModule:
  main()
