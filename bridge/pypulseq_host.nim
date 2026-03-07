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
import nimpulseqgui/io  # makeProtocolPreamble
import nimpy
import std/[os, strutils, parseopt]

proc isPyNone*(o: PyObject): bool =
  ## Returns true if the Python object is ``None``.
  if o.isNil: return true
  let none = pyBuiltinsModule().getAttr("None")
  return o == none

const pythonHome {.strdefine.} = "python"  ## Bundled CPython relative to exe dir.

# ── Marshalling: Nim → Python ──────────────────────────────────────────────

proc optsToPyDict*(opts: Opts): PyObject =
  ## Converts scanner ``Opts`` to a Python dict consumable by the plugin.
  let builtins = pyBuiltinsModule()
  result = builtins.callMethod("dict")
  result["maxGrad"] = opts.maxGrad
  result["maxSlew"] = opts.maxSlew
  result["gradRasterTime"] = opts.gradRasterTime
  result["rfDeadTime"] = opts.rfDeadTime
  result["rfRingdownTime"] = opts.rfRingdownTime
  result["adcDeadTime"] = opts.adcDeadTime
  result["adcRasterTime"] = opts.adcRasterTime
  result["rfRasterTime"] = opts.rfRasterTime
  result["blockDurationRaster"] = opts.blockDurationRaster
  result["B0"] = opts.B0
  result["gamma"] = opts.gamma

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

# ── Callback wrappers ─────────────────────────────────────────────────────

proc wrapGetDefaultProtocol*(plugin: PyPlugin): ProcGetDefaultProtocol =
  ## Wraps ``plugin.get_default_protocol(opts_dict) → dict`` as ``ProcGetDefaultProtocol``.
  return proc(opts: Opts): MRProtocolRef =
    let pyResult = plugin.module.callMethod("get_default_protocol", optsToPyDict(opts))
    return pyDictToProt(pyResult)

proc wrapValidateRich*(plugin: PyPlugin): ProcValidateRich =
  ## Wraps ``plugin.validate_protocol(opts_dict, prot_dict) → dict`` as ``ProcValidateRich``.
  ## Returns the full ``ValidationResult{valid, duration, info}``.
  return proc(opts: Opts, prot: MRProtocolRef): ValidationResult =
    let pyResult = plugin.module.callMethod("validate_protocol",
                                             optsToPyDict(opts), protToPyDict(prot))
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

proc callMakeSequenceString*(plugin: PyPlugin, opts: Opts, prot: MRProtocolRef): string =
  ## Calls Python ``make_sequence`` and returns the ``.seq`` file content as a string.
  ## The caller is responsible for writing it to disk.
  let pyResult = plugin.module.callMethod("make_sequence",
                                           optsToPyDict(opts), protToPyDict(prot))
  return pyResult.to(string)

proc wrapMakeSequence*(plugin: PyPlugin): ProcMakeSequence =
  ## Wraps ``plugin.make_sequence`` as nimpulseqgui's ``ProcMakeSequence``.
  ##
  ## In both the GUI and GE headless paths, the end goal is the same: write a
  ## ``.seq`` file to disk. The Python plugin returns the file content as a string;
  ## for headless modes ``callMakeSequenceString`` is used directly.
  ##
  ## For the GUI path, nimpulseqgui expects ``ProcMakeSequence`` to return a
  ## ``Sequence`` object (nimpulseq in-memory representation) so it can call
  ## ``writeSeq``. Since nimpulseq is write-only (no ``readSeq`` to parse a
  ## ``.seq`` string back into a ``Sequence``), this wrapper cannot yet produce
  ## the required return type. Blocked on nimpulseq adding ``readSeq``.
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
  # Init bundled CPython location
  if dirExists(getAppDir() / pythonHome):
    putEnv("PYTHONHOME", getAppDir() / pythonHome)

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

  let plugin = loadPyPlugin(scriptPath)
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
    var prot = getDefault(opts)
    if not validateBool(opts, prot):
      echo "FATAL ERROR: default protocol is not valid!"
      quit(1)

    # Apply inline protocol if given
    if inputProtocolString.len > 0:
      let warnings = readProtocolFromString(inputProtocolString, opts, prot, validateBool)
      for w in warnings: echo w

    if listProtocol:
      echo "PROTOCOL"
      echo makeProtocolPreamble(prot)
      quit(0)

    if validateOnly:
      let vr = validateRich(opts, prot)
      echo formatValidationJson(vr)
      quit(if vr.valid: 0 else: 1)

    if persistentMode:
      # Persistent stdin/stdout command loop.
      while true:
        let cmd = stdin.readLine().strip()
        if cmd == "QUIT" or cmd == "":
          break
        elif cmd == "LIST_PROTOCOL":
          echo "PROTOCOL"
          echo makeProtocolPreamble(prot)
          flushFile(stdout)
        elif cmd == "VALIDATE":
          let preambleStr = readPreambleFromStdin()
          var localProt = prot.copy
          discard readProtocolFromString(preambleStr, opts, localProt, validateBool)
          let vr = validateRich(opts, localProt)
          echo formatValidationPlain(vr)
          flushFile(stdout)
        elif cmd.startsWith("GENERATE "):
          let outPath = cmd[9..^1].strip()
          let preambleStr = readPreambleFromStdin()
          var localProt = prot.copy
          discard readProtocolFromString(preambleStr, opts, localProt, validateBool)
          try:
            let seqContent = callMakeSequenceString(plugin, opts, localProt)
            let preamble = makeProtocolPreamble(localProt)
            writeFile(outPath, preamble & "\n" & seqContent)
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
                  "PyPulseq: " & scriptPath.extractFilename())

when isMainModule:
  main()
