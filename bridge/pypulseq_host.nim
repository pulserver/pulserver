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
import nimpulseqgui/io  # makeProtocolPreamble (GUI path only)
import nimpy
import nimpy/py_lib as nimpy_lib
import std/[dynlib, os, strutils, parseopt]

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
  ## Strategy: parse ``pyvenv.cfg``'s ``home`` key (the bin dir of the base
  ## Python installation), then search its sibling ``lib/`` directory for the
  ## highest-versioned ``libpython3.*.so*`` file.
  let cfgPath = venvHome / "pyvenv.cfg"
  if not fileExists(cfgPath):
    return ("", "")
  var pythonBinDir = ""
  for line in lines(cfgPath):
    let stripped = line.strip()
    if stripped.startsWith("home ") or stripped.startsWith("home="):
      let parts = stripped.split('=', maxsplit = 1)
      if parts.len == 2 and parts[0].strip() == "home":
        pythonBinDir = parts[1].strip()
        break
  if pythonBinDir.len == 0:
    return ("", "")
  # The lib dir is typically <prefix>/lib where prefix = parentDir(bindir).
  let libDir = parentDir(pythonBinDir) / "lib"
  if not dirExists(libDir):
    return ("", "")
  # Prefer the most specific versioned .so first (e.g. libpython3.13.so.1.0).
  for v in ["3.13", "3.12", "3.11", "3.10", "3.9", "3.8"]:
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
  var forwardArgs: seq[string] = @[]  # args to forward to makeSequenceExe
  var expectScriptPath = false
  var expectProtocolString = false

  for kind, key, val in getopt():
    case kind
    of cmdArgument:
      if expectScriptPath:
        scriptPath = key
        expectScriptPath = false
      elif expectProtocolString:
        inputProtocolString = key
        expectProtocolString = false
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
    # Build opts from forwarded flags. We re-use nimpulseqgui's Opts builder
    # by calling through the same pipeline, but for headless we need the
    # protocol + opts ourselves. Parse just --manufacturer/--model/--B0 etc.
    let opts = newOpts()  # TODO: parse forwarded hardware flags for full fidelity
    let defaultProt = getDefault(opts)
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
      echo makeProtocolPreamble(defaultProt)
      quit(0)

    if persistentMode:
      # Stateless stdin/stdout command loop.
      # Each command starts from defaultProt (the plugin's fixed schema).
      # No state is carried between commands — persistence is purely about
      # keeping the Python interpreter warm.
      while true:
        let cmd = stdin.readLine().strip()
        if cmd == "QUIT" or cmd == "":
          break
        elif cmd == "LIST_PROTOCOL":
          echo "PROTOCOL"
          echo makeProtocolPreamble(defaultProt)
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
