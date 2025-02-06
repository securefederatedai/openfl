{% set truncated_fullname = fullname.split('.')[-1] %}
{% if truncated_fullname | length < 4 %}
   {% set truncated_fullname = truncated_fullname.ljust(4) %}
{% endif %}
Exception - {{ truncated_fullname | escape | underline}}
 
 
.. currentmodule:: {{ module }}
 
.. autofunction:: {{ objname }}
 
   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}
 
   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}