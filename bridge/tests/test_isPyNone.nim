## Quick test that isPyNone works for Python None objects.
import nimpy
import ../pypulseq_host
import nimpulseqgui/definitions

let builtins = pyBuiltinsModule()
let none = builtins.getAttr("None")
let notNone = builtins.callMethod("int", 42)

doAssert isPyNone(none), "Python None should be detected as None"
doAssert not isPyNone(notNone), "Python int(42) should not be None"

# Test via the actual plugin
let plugin = loadPyPlugin("tests/test_plugin.py")
let validateRich = wrapValidateRich(plugin)
let opts = newOpts()
var prot = wrapGetDefaultProtocol(plugin)(opts)

# Make protocol invalid: set TE > TR
prot["TE"].floatVal = 50.0  # TE > TR → invalid

let vr = validateRich(opts, prot)
doAssert not vr.valid, "Should be invalid when TE > TR"
doAssert vr.duration == -1.0, "Duration should be -1.0 for None, got: " & $vr.duration
doAssert vr.info == "TE must be < TR", "Info should have error message, got: " & vr.info

echo "All isPyNone tests passed!"
