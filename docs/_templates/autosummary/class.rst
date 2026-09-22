{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

{% set public = methods | reject("eq", "__init__") | list %}
{% if public %}
.. rubric:: {{ _("Methods") }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in public %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
.. rubric:: {{ _("Attributes") }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
