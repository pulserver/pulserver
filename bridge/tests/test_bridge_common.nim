## Tests for bridge_common — ValidationResult, protocol parsing, formatting.
##
## Run with:  nim r tests/test_bridge_common.nim
## Or via:    nimble test

import std/unittest
import std/strutils
import ../bridge_common
import nimpulseqgui/io  # makeProtocolPreamble for round-trip test

# ── Test helpers ───────────────────────────────────────────────────────────

proc makeTestProtocol(): MRProtocolRef =
  ## Creates a small protocol for testing, covering all property types.
  result = newProtocol()
  result["D_Header"] = newDescriptionProperty("Test Protocol")
  result["TE"] = newFloatProperty(5.0, 1.0, 100.0, 0.1, validate=pvDoSearch, unit="ms")
  result["TR"] = newFloatProperty(10.0, 1.0, 1000.0, 0.1, unit="ms")
  result["FOV"] = newIntProperty(256, 64, 512, 1, unit="mm")
  result["Matrix"] = newIntProperty(256, 32, 512, 32)
  result["FatSat"] = newBoolProperty(false)
  result["Mode"] = newStringListProperty("Fast", @["Fast", "Normal", "High"])

proc alwaysValid(opts: Opts, prot: MRProtocolRef): bool =
  return true

proc teRangeValidator(opts: Opts, prot: MRProtocolRef): bool =
  ## Rejects if TE < 2.0 ms.
  return prot["TE"].floatVal >= 2.0

# ── ValidationResult ──────────────────────────────────────────────────────

suite "ValidationResult":
  test "toBool converter — valid":
    let vr = ValidationResult(valid: true, duration: 5.0, info: "TA = 5.0 s")
    check vr == true          # implicit converter
    check vr.valid == true

  test "toBool converter — invalid":
    let vr = ValidationResult(valid: false, duration: -1.0, info: "TE too short")
    check vr == false
    check vr.valid == false

  test "default fields":
    var vr: ValidationResult
    check vr.valid == false
    check vr.duration == 0.0
    check vr.info == ""

# ── formatValidationJson ──────────────────────────────────────────────────

suite "formatValidationJson":
  test "valid with duration and info":
    let vr = ValidationResult(valid: true, duration: 5.32, info: "TA = 5.32 s")
    let json = formatValidationJson(vr)
    check "\"valid\": true" in json
    check "\"duration\": 5.32" in json
    check "\"info\": \"TA = 5.32 s\"" in json

  test "invalid with no duration":
    let vr = ValidationResult(valid: false, duration: -1.0, info: "TE too short")
    let json = formatValidationJson(vr)
    check "\"valid\": false" in json
    check "\"duration\": null" in json
    check "\"info\": \"TE too short\"" in json

  test "valid with no info":
    let vr = ValidationResult(valid: true, duration: 3.0, info: "")
    let json = formatValidationJson(vr)
    check "\"info\": null" in json

# ── formatValidationPlain ─────────────────────────────────────────────────

suite "formatValidationPlain":
  test "valid with duration and info":
    let vr = ValidationResult(valid: true, duration: 5.32, info: "TA = 5.32 s")
    let line = formatValidationPlain(vr)
    check line.startsWith("VALID ")
    check "5.32" in line
    check "TA = 5.32 s" in line

  test "invalid with info":
    let vr = ValidationResult(valid: false, duration: -1.0, info: "TE too short")
    let line = formatValidationPlain(vr)
    check line.startsWith("INVALID ")
    check "TE too short" in line

  test "invalid without info":
    let vr = ValidationResult(valid: false, duration: -1.0, info: "")
    let line = formatValidationPlain(vr)
    check "validation failed" in line

  test "valid with unknown duration":
    let vr = ValidationResult(valid: true, duration: -1.0, info: "")
    let line = formatValidationPlain(vr)
    check line.startsWith("VALID ?")

# ── readProtocolFromString ────────────────────────────────────────────────

suite "readProtocolFromString":
  test "parses valid preamble":
    let prot = makeTestProtocol()
    let opts = newOpts()
    let preamble = """
[NimPulseqGUI Protocol]
TE: 7.5
TR: 20.0
FOV: 128
Matrix: 64
FatSat: True
Mode: Normal
[NimPulseqGUI Protocol End]
"""
    let warnings = readProtocolFromString(preamble, opts, prot, alwaysValid)
    check warnings.len == 0
    check prot["TE"].floatVal == 7.5
    check prot["TR"].floatVal == 20.0
    check prot["FOV"].intVal == 128
    check prot["Matrix"].intVal == 64
    check prot["FatSat"].boolVal == true
    check prot["Mode"].stringVal == "Normal"

  test "parses preamble with comment-prefixed lines":
    let prot = makeTestProtocol()
    let opts = newOpts()
    let preamble = """
# [NimPulseqGUI Protocol]
# TE: 8.0
# [NimPulseqGUI Protocol End]
"""
    let warnings = readProtocolFromString(preamble, opts, prot, alwaysValid)
    check warnings.len == 0
    check prot["TE"].floatVal == 8.0

  test "warns on unknown parameter":
    let prot = makeTestProtocol()
    let opts = newOpts()
    let preamble = """
[NimPulseqGUI Protocol]
TE: 5.0
UnknownParam: 42
[NimPulseqGUI Protocol End]
"""
    let warnings = readProtocolFromString(preamble, opts, prot, alwaysValid)
    check warnings.len == 1
    check "UnknownParam" in warnings[0]

  test "warns on invalid stringlist value":
    let prot = makeTestProtocol()
    let opts = newOpts()
    let preamble = """
[NimPulseqGUI Protocol]
Mode: Turbo
[NimPulseqGUI Protocol End]
"""
    let warnings = readProtocolFromString(preamble, opts, prot, alwaysValid)
    check warnings.len == 1
    check "Turbo" in warnings[0]
    check prot["Mode"].stringVal == "Fast"  # unchanged from default

  test "warns when no preamble markers found":
    let prot = makeTestProtocol()
    let opts = newOpts()
    let warnings = readProtocolFromString("just some text", opts, prot, alwaysValid)
    check warnings.len >= 1
    check "No protocol preamble found" in warnings[0]

  test "reverts to default when validation fails":
    let prot = makeTestProtocol()
    let opts = newOpts()
    let preamble = """
[NimPulseqGUI Protocol]
TE: 0.5
[NimPulseqGUI Protocol End]
"""
    let warnings = readProtocolFromString(preamble, opts, prot, teRangeValidator)
    check warnings.len >= 1
    check "did not pass validation" in warnings[^1]
    check prot["TE"].floatVal == 5.0  # default preserved

  test "stops at [VERSION] marker":
    let prot = makeTestProtocol()
    let opts = newOpts()
    let preamble = """
[NimPulseqGUI Protocol]
TE: 9.0
[VERSION] 1.4.0
FOV: 100
[NimPulseqGUI Protocol End]
"""
    let warnings = readProtocolFromString(preamble, opts, prot, alwaysValid)
    check prot["TE"].floatVal == 9.0
    check prot["FOV"].intVal == 256  # not updated — past [VERSION]

  test "handles description with escaped newlines":
    let prot = makeTestProtocol()
    let opts = newOpts()
    let preamble = """
[NimPulseqGUI Protocol]
D_Header: Line1\nLine2\nLine3
[NimPulseqGUI Protocol End]
"""
    let warnings = readProtocolFromString(preamble, opts, prot, alwaysValid)
    check warnings.len == 0
    check prot["D_Header"].description == "Line1\nLine2\nLine3"

  test "handles bool false values":
    let prot = makeTestProtocol()
    prot["FatSat"].boolVal = true
    let opts = newOpts()
    let preamble = """
[NimPulseqGUI Protocol]
FatSat: False
[NimPulseqGUI Protocol End]
"""
    let warnings = readProtocolFromString(preamble, opts, prot, alwaysValid)
    check warnings.len == 0
    check prot["FatSat"].boolVal == false

  test "skips blank lines inside preamble":
    let prot = makeTestProtocol()
    let opts = newOpts()
    let preamble = """
[NimPulseqGUI Protocol]
TE: 6.0

TR: 15.0
[NimPulseqGUI Protocol End]
"""
    let warnings = readProtocolFromString(preamble, opts, prot, alwaysValid)
    check warnings.len == 0
    check prot["TE"].floatVal == 6.0
    check prot["TR"].floatVal == 15.0

# ── Preamble constants ────────────────────────────────────────────────────

suite "preamble constants":
  test "start marker matches nimpulseqgui format":
    check protocolPreambleStart == "[NimPulseqGUI Protocol]"

  test "end marker matches nimpulseqgui format":
    check protocolPreambleEnd == "[NimPulseqGUI Protocol End]"

# ── Round-trip: makeProtocolPreamble → readProtocolFromString ─────────────

suite "preamble round-trip":
  test "serialize then parse preserves all values":
    let prot = makeTestProtocol()
    prot["TE"].floatVal = 12.3
    prot["FOV"].intVal = 192
    prot["FatSat"].boolVal = true
    prot["Mode"].stringVal = "High"
    let opts = newOpts()
    let preamble = makeProtocolPreamble(prot)

    # Parse into a fresh protocol with the same structure
    let prot2 = makeTestProtocol()
    let warnings = readProtocolFromString(preamble, opts, prot2, alwaysValid)
    check warnings.len == 0
    check prot2["TE"].floatVal == 12.3
    check prot2["FOV"].intVal == 192
    check prot2["FatSat"].boolVal == true
    check prot2["Mode"].stringVal == "High"
