#!/bin/sh
# The image's entry point. Without arguments, or with options alone, it serves
# the console of the virtual scanner with those options after its own: an
# option of one value takes the value given last, and --plugins,
# --recon-plugins and --origin add to the image's. Any other pulserver command
# runs in the console's place.
case "${1:-}" in
"" | -*)
    set -- console \
        --plugins=/console/user/plugins --plugins=/console/plugins \
        --limits=/console/limits.txt --store=/console/designs \
        --fields=/console/fields \
        --recon-plugins=/console/user/recon --recon-plugins=/console/recon \
        --host=0.0.0.0 --port=8765 --speed=1 \
        --origin=https://pulserver.github.io --origin=http://localhost:8000 \
        --origin=http://127.0.0.1:8000 \
        "$@"
    ;;
esac
exec pulserver "$@"
