#!/bin/bash
P="after_ck.sh h3 6""00"; for p in $(pgrep -u $USER -f "$P"); do kill $p && echo killed $p; done
