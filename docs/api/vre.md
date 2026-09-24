# Reconstruction proxy

The reconstruction computer's service: it receives each series from the
scanner's reconstruction client, enriches it from the sequence that played it,
and runs it on a reconstruction worker.

```{eval-rst}
.. currentmodule:: pulserver.vre
```

The proxy listens on a TCP port and is started with

```bash
python -m pulserver.vre --store DIR --port N --plugins DIR [--slots N] [--spares 1]
```

The MRD header of a series names the design it was played from; the proxy
reads that design's sequence chain and fills in the header and every
acquisition. The readouts arrive demodulated to the prescribed field-of-view
centre by the playout, and are passed on as received. Routing, slots and the
queue are described in {doc}`../explanations/reconstruction`, and the messages
and fields of a series in {doc}`../user-guide/reconstruction-client`.

## Proxy

| Object | Description |
| --- | --- |
| {obj}`~pulserver.vre.ReconProxy` | TCP MRD server routing each series to a reconstruction worker. |

## Designs

| Object | Description |
| --- | --- |
| {obj}`~pulserver.vre.DesignCache` | The designs of a store, the most recently read kept tabulated. |
| {obj}`~pulserver.vre.Design` | A stored design as the reconstruction side reads it. |
| {obj}`~pulserver.vre.DESIGN_PARAMETER` | Header user parameter naming the design a series was played from. |

## Enrichment

| Object | Description |
| --- | --- |
| {obj}`~pulserver.vre.SequenceTable` | The readouts of a sequence chain as MRD describes them, in play order. |
| {obj}`~pulserver.vre.TableSpace` | One encoding space of a sequence table. |
| {obj}`~pulserver.vre.enrich_header` | Describe the table's encoding spaces and sequence parameters in an MRD header. |
| {obj}`~pulserver.vre.enrich_acquisition` | Apply one row of the table to an acquisition, in place. |
