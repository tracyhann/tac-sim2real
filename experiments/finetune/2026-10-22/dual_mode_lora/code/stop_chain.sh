#!/bin/bash
# stop the dual-mode chain waiter (its loops and pool steps keep running); also kill its sleep so bash exits at once
for pid in $(pgrep -u $USER -f "lora_ge/chain_ge[.]sh"); do echo "kill chain $pid"; kill $pid; pkill -u $USER -P $pid sleep; done
