# Reconstruction proxy

The reconstruction computer's service: it receives each series from the
scanner's reconstruction client, enriches it from the sequence that played it,
and runs it on a reconstruction worker.

```{eval-rst}
.. currentmodule:: pulserver.vre
```

The proxy listens on a TCP port and is started with

```bash
python -m pulserver.vre --base DIR --port N --plugins DIR [--slots N] [--spares 1]
```

The MRD header of a series names the revision it was played from; the proxy
reads that revision's sequence chain, fills in the header and every
acquisition, and demodulates the readouts to the prescription centre. Routing,
slots and the queue are described in {doc}`../explanations/reconstruction`.

## Proxy

| Object | Description |
| --- | --- |
| {obj}`~pulserver.vre.ReconProxy` | TCP MRD server routing each series to a reconstruction worker. |

## Revisions

| Object | Description |
| --- | --- |
| {obj}`~pulserver.vre.RevisionStore` | The revisions under `<base>/bucket/`, each read once and kept. |
| {obj}`~pulserver.vre.Revision` | A generated design as the reconstruction side reads it. |
| {obj}`~pulserver.vre.SESSION_PARAMETER` | Header user parameter naming the session a series was played from. |
| {obj}`~pulserver.vre.REVISION_PARAMETER` | Header user parameter naming the revision a series was played from. |

## Enrichment

| Object | Description |
| --- | --- |
| {obj}`~pulserver.vre.SequenceTable` | The readouts of a sequence chain as MRD describes them, in play order. |
| {obj}`~pulserver.vre.TableSpace` | One encoding space of a sequence table. |
| {obj}`~pulserver.vre.enrich_header` | Describe the table's encoding spaces and sequence parameters in an MRD header. |
| {obj}`~pulserver.vre.enrich_acquisition` | Apply one row of the table to an acquisition, in place. |
| {obj}`~pulserver.vre.fov_offset_m` | Prescription centre a header requests readouts to be demodulated to, in metres. |
| {obj}`~pulserver.vre.FOV_OFFSET_PARAMETER` | Header user parameter holding the prescription centre, in mm along the gradient axes. |
