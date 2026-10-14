#!/usr/bin/env python3
"""X1 in-sandbox launcher (runs as lu-sandbox inside bwrap). Applies rlimits +
a seccomp denylist (defense-in-depth; network is already gone via the empty
netns), then executes the candidate program. stdout/stderr are the candidate's."""
import sys, os, resource

def _limit():
    cpu = int(os.environ.get("X1_CPU", "4"))
    fsize = int(os.environ.get("X1_FSIZE", str(16 * 1024 * 1024)))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    try: resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except Exception: pass
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

DENY = ["socket","socketpair","connect","bind","listen","accept","accept4",
        "sendto","recvfrom","sendmsg","recvmsg","getpeername",
        "ptrace","process_vm_readv","process_vm_writev",
        "mount","umount2","pivot_root","chroot","mount_setattr",
        "unshare","setns","keyctl","add_key","request_key","bpf",
        "init_module","finit_module","delete_module","kexec_load",
        "open_by_handle_at","reboot","swapon","swapoff","settimeofday"]

def _seccomp():
    try:
        import seccomp
        f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
        try: f.set_attr(seccomp.Attr.CTL_NNP, 1)
        except Exception: pass
        for sc in DENY:
            try: f.add_rule(seccomp.KILL, sc)
            except Exception: pass
        f.load()
        return "loaded"
    except Exception as e:
        return "UNAVAILABLE:%s" % e

def main():
    _limit()
    status = _seccomp()
    # Record seccomp status to a sidecar the broker can read (not stdout).
    try:
        with open("/work/.x1_seccomp", "w") as fh: fh.write(status)
    except Exception: pass
    prog = sys.argv[1]
    with open(prog) as fh: src = fh.read()
    sys.argv = [prog]
    g = {"__name__": "__main__", "__file__": prog}
    exec(compile(src, prog, "exec"), g)

if __name__ == "__main__":
    main()
