{{/*
Expand the name of the chart.
*/}}
{{- define "openfl.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
Default trim is to 63, but let's leave space for "-collaborator-99" suffix, so 63-16=47
If release name contains chart name it will be used as a full name.
*/}}
{{- define "openfl.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 47 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 47 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 47 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "openfl.fullnameAggregator" -}}
{{- include "openfl.fullname" . | trunc 47 | trimSuffix "-" }}-aggregator
{{- end }}

{{- define "openfl.fullnameCollaborator" -}}
{{- include "openfl.fullname" . | trunc 47 | trimSuffix "-" }}-collaborator
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "openfl.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "openfl.labels" -}}
helm.sh/chart: {{ include "openfl.chart" . }}
{{ include "openfl.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "openfl.selectorLabels" -}}
app.kubernetes.io/name: {{ include "openfl.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "openfl.selectorLabelsAggregator" -}}
{{ include "openfl.selectorLabels" .}}
app.kubernetes.io/component: aggregator
{{- end }}

{{- define "openfl.selectorLabelsCollaborator" -}}
{{ include "openfl.selectorLabels" .}}
app.kubernetes.io/component: collaborator
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "openfl.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "openfl.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
