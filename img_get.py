from frames import r_fms, d_fm, BaiLanError


def get_frames(fn_index: int, *args, **kwargs):
    _get_frames = r_fms[fn_index](*args, **kwargs)
    return _get_frames


def get_frames_delay(fn_index: int, fps=30):
    getts = []
    for fmt in d_fm[fn_index]:
        _rfm = str(fmt).split(".")
        fm_sec = int(_rfm[0])

        if not _rfm[1].startswith("0") and len(_rfm[1]) < len(str(fps)):
            fm_frame = int(_rfm[1] + "0")
        else:
            fm_frame = int(_rfm[1])

        tframe_time_s = fm_sec + fm_frame / fps
        getts.append(tframe_time_s)

    retts = []
    last_ts = 0
    for rt in getts:
        if last_ts == 0:
            last_ts = rt
        else:
            _apd = rt - last_ts
            if _apd <= 0:
                _apd = 0.001
            retts.append(_apd)
            last_ts = rt

    print("get_frames_delay", retts)
    return retts

def calc_fdelay_ms(fm_count: int, fps=30):
    ms_perfm = 60 / fps * 1000
    return int(fm_count * ms_perfm)
