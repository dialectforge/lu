# Deploy artifacts (OP-CODE-EXEC Phase 6/7)
lu-codeexec.service: privilege-separated X1 broker (localhost socket).
30-codeexec.conf: guest drop-in granting socket access ONLY; guest hardening (NoNewPrivileges/ProtectSystem=strict) unchanged.
tmpfiles.d/lu-codeexec.conf: creates /run/lu-codeexec.
