# Protocol

`pulserver.protocol`: the entries of a scanner protocol and the text blocks that
carry them between the orchestrator and the interpreter.

```{eval-rst}
.. currentmodule:: pulserver.protocol
```

## Parameters

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Parameter
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: autosummary/enum.rst

   Kind
   InputMode
   TEPreset
   TRPreset
```

## Wire blocks

A listing carries every entry with its schema, as `LIST_PROTOCOL` returns it; a
value block carries values only, as requests and replies do.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   PROTOCOL_BEGIN
   PROTOCOL_END
   format_listing
   parse_listing
   format_values
   parse_values
   Validation
   format_validation
   parse_validation
```
