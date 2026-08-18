## bridge_common — shared types and helpers for pulserver bridge hosts.
##
## This module lives **outside** nimpulseqgui and adds the richer
## ``ValidationResult`` type plus protocol string parsing that the
## GE driver needs, without modifying upstream nimpulseqgui at all.
##
## Both ``pypulseq_host`` and ``matlab_host`` import this module.

import nimpulseqgui
import std/strutils, std/strformat

# Re-export everything the hosts need from nimpulseqgui.
export nimpulseqgui

# ── Rich validation result (superset of nimpulseqgui's bool) ───────────────

type
  ValidationResult* = object
    ## Extended validation outcome used by Python/MATLAB plugins.
    ##
    ## The GE driver (headless modes) sees all three fields.
    ## When delegating to nimpulseqgui's GUI the bridge wraps this
    ## down to a plain ``bool`` via the ``ProcValidateProtocol`` adapter.
    valid*: bool
    duration*: float    ## seconds; < 0 means "not computed"
    info*: string       ## free-form text for display (e.g. "TA = 5.32 s")

  ProcValidateRich* = proc(opts: Opts, protocol: MRProtocolRef): ValidationResult {. closure .}
    ## Callback type for rich validation used by bridge hosts.
    ## Returned by the Python/MATLAB adapter layer.

converter toBool*(vr: ValidationResult): bool =
  ## Allows ``ValidationResult`` to be used wherever ``bool`` is expected.
  vr.valid

# ── Protocol preamble constants ────────────────────────────────────────────

const
  protocolPreambleStart* = "[NimPulseqGUI Protocol]"
  protocolPreambleEnd*   = "[NimPulseqGUI Protocol End]"

# ── Protocol string parsing (complements nimpulseqgui's file-based reader) ─

proc readProtocolFromString*(content: string, opts: Opts, defaultProt: MRProtocolRef,
                              validateProc: ProcValidateProtocol): seq[string] =
  ## Parses protocol parameters from a raw string containing the preamble block.
  ##
  ## Works identically to nimpulseqgui's ``readProtocolFromFile``, but
  ## operates on an in-memory string. The string may contain just the
  ## preamble block or a full ``.seq`` file.
  ##
  ## Returns a sequence of warning strings (empty on full success).
  var protocolFound = false
  var localProt = defaultProt.copy
  var warnings: seq[string] = @[]

  for rawLine in content.splitLines():
    var line = rawLine
    if line.startsWith("#"):
      line = line[1..^1].strip()
    if line.contains("[VERSION]"):
      break
    if line.contains(protocolPreambleStart):
      protocolFound = true
      continue
    if line.contains(protocolPreambleEnd):
      break
    if protocolFound:
      let parts = line.split(": ", maxsplit = 1)
      if parts.len < 2:
        continue
      let varName = parts[0].strip()
      let varValue = parts[1].strip()
      if not localProt.contains(varName):
        warnings.add(&"Warning: Protocol parameter '{varName}' not recognized. Ignoring.")
        continue
      case localProt[varName].pType
      of ptDescription:
        localProt[varName].description = varValue.replace("\\n", "\n")
      of ptInt:
        localProt[varName].intVal = parseInt(varValue)
      of ptFloat:
        localProt[varName].floatVal = parseFloat(varValue)
      of ptBool:
        localProt[varName].boolVal = varValue.toLowerAscii() == "true"
      of ptStringList:
        if localProt[varName].stringList.contains(varValue):
          localProt[varName].stringVal = varValue
        else:
          warnings.add(&"Warning: Value '{varValue}' for '{varName}' not in allowed list. Ignoring.")

  if not protocolFound:
    warnings.add("Warning: No protocol preamble found. Using default protocol values.")
  var validationPassed = false
  try:
    validationPassed = validateProc(opts, localProt)
  except Exception:
    validationPassed = false
  if not validationPassed:
    warnings.add("Warning: Protocol did not pass validation. Using default protocol values.")
    return warnings
  defaultProt[] = localProt[]
  return warnings

# ── Persistent-mode helpers ────────────────────────────────────────────────

proc readPreambleFromStdin*(): string =
  ## Reads preamble lines from stdin until ``protocolPreambleEnd`` is seen.
  var lines: seq[string] = @[]
  while true:
    let line = stdin.readLine()
    lines.add(line)
    if line.contains(protocolPreambleEnd):
      break
  return lines.join("\n")

proc formatValidationJson*(vr: ValidationResult): string =
  ## Serializes a ``ValidationResult`` to a single-line JSON string.
  let durationStr = if vr.duration >= 0.0: $vr.duration else: "null"
  let infoStr = if vr.info.len > 0: "\"" & vr.info & "\"" else: "null"
  return "{\"valid\": " & $vr.valid & ", \"duration\": " & durationStr & ", \"info\": " & infoStr & "}"

proc formatValidationPlain*(vr: ValidationResult): string =
  ## Formats a ``ValidationResult`` as a single plain-text line
  ## for the persistent pipe protocol.
  if vr.valid:
    let info = if vr.info.len > 0: " " & vr.info else: ""
    return "VALID " & (if vr.duration >= 0.0: $vr.duration else: "?") & info
  else:
    return "INVALID " & (if vr.info.len > 0: vr.info else: "validation failed")
