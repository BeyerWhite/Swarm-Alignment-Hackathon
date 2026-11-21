#!/bin/bash
# Couzin study launcher. Its name deliberately does not match any study's "queue" pattern: other studies' wrappers
# wait while a *queue[N].sh runs, so a Couzin queue that waited under such a name could deadlock with them.
# Waits for the pids given, then until no other stage script or paced job runs, then execs the queue.
# Usage: nohup caffeinate -i bash code/couzin_launch.sh <1|2> <ledger-cap> [pid ...] > scratch/couzin_queueN.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
Q=$1; shift
. code/couzin_common.sh "$@" || exit $?
shift
wait_turn "$@"
exec bash "code/couzin_queue$Q.sh" "$CAP"
