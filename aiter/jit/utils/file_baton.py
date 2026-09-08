# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# mypy: allow-untyped-defs
import logging
import multiprocessing
import os
import socket
import time

logger = logging.getLogger("aiter")


class FileBaton:
    """A primitive, file-based synchronization utility.

    The lock file records the owning ``pid`` and host so that a crashed or
    killed builder (which never reaches :meth:`release`) leaves behind a
    *stale* lock that waiters can detect and break, instead of deadlocking
    forever. This also covers the empty/0-byte lock left when a process dies
    between creating and writing the file.
    """

    def __init__(self, lock_file_path, wait_seconds=0.2, stale_grace_seconds=10.0):
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
            stale_grace_seconds: For an orphaned lock with no readable owner
                info (e.g. a 0-byte lock from a crash), how old it must be
                before being treated as stale. Protects the brief window
                between create and write in a healthy builder.
        """
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.stale_grace_seconds = stale_grace_seconds
        self.fd = None

    def try_acquire(self):
        """
        Try to atomically create a file under exclusive access and stamp it
        with the current owner (pid + host) for stale detection.

        Returns:
            True if the file could be created, else False.
        """
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            os.write(self.fd, f"{os.getpid()}\n{socket.gethostname()}\n".encode())
            os.fsync(self.fd)
        except OSError:
            pass
        return True

    def wait(self):
        """
        Periodically sleep until the baton is released by its holder, or break
        the lock if its holder is dead.

        Returns:
            True if the holder released the lock normally (its work is done).
            False if a stale lock was broken — the caller should re-acquire
            and redo the work, since no holder ever finished it.
        """
        logger.info(
            f"[pid={os.getpid()} pname={multiprocessing.current_process().name}] "
            f"waiting for baton release at {self.lock_file_path}"
        )
        while True:
            if not os.path.exists(self.lock_file_path):
                return True
            if self._is_stale() and self._try_break_stale():
                logger.warning(
                    f"[pid={os.getpid()}] broke stale lock at "
                    f"{self.lock_file_path} (dead/abandoned holder)"
                )
                return False
            time.sleep(self.wait_seconds)

    def release(self):
        """Release the baton and remove its file."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            os.remove(self.lock_file_path)
        except FileNotFoundError:
            pass

    # ---- stale-lock detection ----

    def _read_owner(self):
        """Return (pid, host) recorded in the lock file, or (None, None)."""
        try:
            with open(self.lock_file_path, "r") as f:
                lines = f.read().splitlines()
        except (FileNotFoundError, OSError):
            return None, None
        if len(lines) < 2 or not lines[0].strip().isdigit():
            return None, None
        return int(lines[0].strip()), lines[1].strip()

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but owned by another user
        # The pid is still present in the process table, but a zombie
        # ('Z' state, i.e. exited-but-unreaped) has already terminated and
        # will never resume building -- it will only ever be reaped by its
        # parent (or, once orphaned, by PID 1). In containers where PID 1 is
        # not a real init and never reaps orphans (observed here: a `ninja`
        # build helper left as a permanent `[ninja] <defunct>` zombie under
        # ppid=1), os.kill(pid, 0) keeps succeeding forever, so a lock whose
        # recorded owner died mid-build would never be judged stale and
        # every waiter would spin in wait() indefinitely -- exactly the
        # "JIT build stuck / kernel_agent silent" failure mode this baton
        # exists to prevent. Treat a zombie as dead so its lock can be
        # broken and the work retried.
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                stat = f.read().decode(errors="replace")
            # comm (2nd field) is parenthesized and may itself contain
            # spaces/parens, so find the state field after the last ')'.
            state = stat[stat.rindex(")") + 2]
            if state == "Z":
                return False
        except (FileNotFoundError, OSError, ValueError, IndexError):
            # /proc unavailable or unparsable: fall back to the kill(0)
            # verdict rather than risk mis-detecting an unrelated pid.
            pass
        return True

    def _is_stale(self):
        """A lock is stale if its recorded holder is dead, or if it carries no
        owner info and has outlived the grace period (orphaned/0-byte lock)."""
        pid, host = self._read_owner()
        if pid is None:
            # No readable owner: only trust mtime, and only for our own host
            # cannot be verified, so fall back to an age-based grace window.
            try:
                age = time.time() - os.path.getmtime(self.lock_file_path)
            except OSError:
                return False
            return age > self.stale_grace_seconds
        if host != socket.gethostname():
            # Different host (e.g. shared filesystem): can't check liveness,
            # never steal — avoid breaking a live remote builder's lock.
            return False
        return not self._pid_alive(pid)

    def _try_break_stale(self):
        """Atomically break a stale lock. A secondary ``.steal`` lock ensures
        only one waiter removes it; the rest just loop and re-check."""
        steal_path = self.lock_file_path + ".steal"
        try:
            sfd = os.open(steal_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            # Re-verify under the steal lock to avoid racing a fresh acquire.
            if os.path.exists(self.lock_file_path) and self._is_stale():
                try:
                    os.remove(self.lock_file_path)
                except FileNotFoundError:
                    pass
                return True
            return False
        finally:
            os.close(sfd)
            try:
                os.remove(steal_path)
            except FileNotFoundError:
                pass
