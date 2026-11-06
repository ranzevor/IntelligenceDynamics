import errno


def create_symlink(testcase, link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        unsupported = {errno.EPERM, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}
        if getattr(exc, "winerror", None) == 1314 or exc.errno in unsupported:
            testcase.skipTest(f"symbolic links unavailable: {exc}")
        raise


def create_junction(link, target):
    import _winapi

    _winapi.CreateJunction(str(target), str(link))
