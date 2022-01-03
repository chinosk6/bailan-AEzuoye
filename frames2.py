from PIL import Image, ImageDraw
from typing import List, Callable


class BaiLanError(Exception):
    pass


def paste_image(pt, im, x, y, w=-1, h=-1, with_mask=True):
    w = im.width if w == -1 else w
    h = im.height if h == -1 else h
    im = im.resize((w, h))
    pt.paste(im, (x, y, x + w, y + h), im.convert("RGBA") if with_mask else None)


def paste_image_return(im1, im, x, y, w=-1, h=-1, with_mask=True, fill=(255, 255, 255), x_mirror=False, y_mirror=False):
    w = im.width if w == -1 else w
    h = im.height if h == -1 else h
    pt = Image.new("RGBA", (w, h), fill)
    paste_image(pt, im1, x, y, w, h, with_mask)

    if x_mirror:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if y_mirror:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

    im = im.resize((w, h))
    pt.paste(im, (x, y, x + w, y + h), im.convert("RGBA") if with_mask else None)
    return pt


def generate_turn_around_frames(img: Image, rotation_angle: int, total_frames: int, move_path: List[List],
                                size=(1920, 1080), color=(255, 255, 255), wh=(-1, -1), bgimg=None):
    """
    旋转路径, 注意, 要能整除
    :param img: 图片
    :param rotation_angle: 旋转角度
    :param total_frames: 总帧数
    :param move_path: 移动路径
    :param size: 底图size
    :param color: 底图颜色
    :param wh: 图片长宽
    :param bgimg: 自定义bg
    :return:
    """
    pt = Image.new("RGBA", size, color) if bgimg is None else bgimg.copy()
    draw = ImageDraw.Draw(pt)
    ret = []
    if len(move_path) == 1:
        _move_path = move_path * total_frames
    else:
        if len(move_path) <= 2:
            move_path.reverse()
        path_num = len(move_path)  # 点数
        path_line_count = path_num - 1  # 线条数
        frame_count_per_line = int(total_frames / path_line_count)  # 每条线点数

        _move_path = []
        for _l in range(path_line_count):
            if _l + 1 > len(move_path) - 1:
                break
            start_point = move_path[_l]
            end_point = move_path[_l + 1]
            for f in range(frame_count_per_line):
                pos = f / frame_count_per_line  # 位置百分比
                p_x = pos * start_point[0] + (1 - pos) * end_point[0]
                p_y = pos * start_point[1] + (1 - pos) * end_point[1]
                _move_path.append([p_x, p_y])

    rs = 0
    im = img.copy()
    for n in range(len(_move_path)):
        _path = _move_path[n]

        paste_image(pt, im.rotate(rs), int(_path[0]), int(_path[1]), wh[0], wh[1])
        ret.append(pt.copy())
        if bgimg is None:
            draw.rectangle(xy=(0, 0, size[0], size[1]), fill=color)
        else:
            pt = bgimg.copy()

        rs += rotation_angle
        if rs > 360:
            rs -= 360
    return ret


def get_times_continuous(total_frames: int, time_start: str, time_end: str, fps=30):
    """
    获取连续时间
    :param total_frames: 总帧数
    :param time_start: 开始时间
    :param time_end: 结束时间
    :param fps: fps
    :return:
    """
    ret = []
    starts = time_start.split(".")
    ends = time_end.split(".")
    start_min = int(starts[0])  # 开始分钟
    start_fm = int(starts[1])  # 开始帧数
    end_min = int(ends[0])  # 结束分钟
    end_fm = int(ends[1])  # 结束帧数

    past_min = end_min - start_min  # 经过分钟
    past_fm = end_fm - start_fm
    while past_fm < 0:
        past_min -= 1
        past_fm += fps

    total_past_fm = past_min * fps + past_fm  # 总帧数
    pt_perfm = total_frames / total_past_fm  # 步进

    for n in range(total_past_fm):
        if n == 0 or n == total_past_fm - 1:
            ret.append(float(f"{start_min}.{'%.2d' % int(start_fm)}"))
        else:
            start_fm += pt_perfm
            while start_fm >= fps:
                start_min += 1
                start_fm -= fps
            ret.append(float(f"{start_min}.{'%.2d' % int(start_fm)}"))

    return ret


class BaiLanFrames:  # 帧生成
    def __init__(self):
        self.pt = Image.new("RGBA", (1920, 1080), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.pt)

    def clear_pt(self, fill=(255, 255, 255)):
        self.draw.rectangle(xy=(0, 0, 1920, 1080), fill=fill)

    def scene_template(self):
        ret_frames = []

        pt = self.pt
        im = Image.open("./src/fbg.png")
        paste_image(pt, im, 712, 294, 552, 552)

        ret_frames.append(pt.copy())

        self.clear_pt()
        return ret_frames

    def scene_1(self):  # 三个福
        ret_frames = []

        pt = self.pt
        im = Image.open("./src/fbg.png")
        paste_image(pt, im, 684, 264, 552, 552)  # 正中
        ret_frames.append(pt.copy())

        paste_image(pt, im, 1296, 264, 552, 552)  # 右边
        ret_frames.append(pt.copy())

        paste_image(pt, im, 72, 264, 552, 552)  # 左边
        ret_frames.append(pt.copy())

        paste_image(pt, im, 712, 294, 552, 552)  # 左边
        ret_frames.append(pt.copy())

        self.clear_pt()
        return ret_frames

    def scene_2(self):
        global d_fm

        ret_frames = []

        pt = self.pt
        im = Image.open("./src/fbg.png")
        paste_image(pt, im, 712, 294, 552, 552)
        ret_frames.append(pt.copy())

        self.clear_pt()
        paste_image(pt, im, 430, 10, 1060, 1060)  # 单大福
        ret_frames.append(pt.copy())

        self.clear_pt()
        paste_image(pt, im, 0, 15, 1060, 1060)  # 双大福
        paste_image(pt, im, 960, 5, 1060, 1060)  # 双大福
        ret_frames.append(pt.copy())

        self.clear_pt()  # 3.19
        paste_image(pt, im, 430, 10, 1060, 1060)  # 单大福
        ret_frames.append(pt.copy())

        self.clear_pt()  # 4.02
        paste_image(pt, im.rotate(180), 430, 10, 1060, 1060)  # 倒单大福
        ret_frames.append(pt.copy())

        fzs = generate_turn_around_frames(im, 47, 63, [[430, 10]], wh=(1060, 1060))  # 旋转单福
        ftimes = get_times_continuous(63, "4.09", "6.11")
        ret_frames = ret_frames + fzs
        for _t in ftimes:
            d_fm[1].append(_t)

        fzs = []
        # fzs_left = generate_turn_around_frames(im, 47, 79, [[0, 15]], wh=(1060, 1060), color=(0, 0, 0, 0))  # 旋转双福
        # for n in range(len(fzs_left)):
        #    fzs_left[n].save(f"./temp/scene2.0/{n}.png")
        # fzs_left = None

        fzs_right = generate_turn_around_frames(im.rotate(180), 47, 79, [[960, 5]], wh=(1060, 1060), color=(0, 0, 0, 0))
        for i in range(len(fzs_right)):
            _pst = paste_image_return(fzs_right[i], Image.open(f"./temp/scene2.0/{i}.png"), 0, 0)
            _pst.convert("RGB").save(f"./temp/scene_2/{i}.jpg")

        for i in range(len(fzs_right)):
            _pst = Image.open(f"./temp/scene_2/{i}.jpg")
            fzs.append(_pst)

        ftimes = get_times_continuous(63, "6.12", "9.01")
        ret_frames = ret_frames + fzs
        for _t in ftimes:
            d_fm[1].append(_t)

        self.clear_pt()
        return ret_frames

    def scene_3(self):
        pt = self.pt
        '''
        self.clear_pt((255, 255, 255, 0))
        im = Image.open("./src/鞭炮.png")

        paste_image(pt, im, 295, 157, 277, 765)
        w_msk = Image.open("./src/白遮罩.png")
        fzl = generate_turn_around_frames(w_msk, 0, 28, [[255, 923], [218, 188]],
                                          wh=(1060, 1060), color=(0, 0, 0, 0), bgimg=pt)  # 9.01 - 9.29, 左边鞭炮

        self.clear_pt((255, 255, 255, 0))
        paste_image(pt, im, 1239, 157, 277, 765)
        fzr = generate_turn_around_frames(w_msk, 0, 26, [[1162, 923], [1162, 188]],
                                          wh=(1060, 1060), color=(0, 0, 0, 0), bgimg=pt)  # 9.23 - 10.19, 右边鞭炮

        self.clear_pt((255, 255, 255, 0))
        paste_image(pt, im, 865, 157, 277, 765)
        fzm = generate_turn_around_frames(w_msk, 0, 26, [[788, 923], [788, 188]],
                                          wh=(1060, 1060), color=(0, 0, 0, 0), bgimg=pt)  # 10.16 - 11.12, 中间鞭炮

        self.clear_pt((255, 255, 255, 0))
        im = Image.open("./src/灯笼.png")
        fzdl = generate_turn_around_frames(im, 0, 60, [[0, 84], [1247, 84]],
                                           wh=(672, 996), color=(0, 0, 0, 0))  # 10.23 - 12.23, 平移灯笼
        self.clear_pt()
        '''
        pt_fms = []
        max_frame = 112
        '''
        _fire = Image.open("./src/火花2.png")
        for n in range(max_frame):
            if n <= len(fzl) - 1:
                paste_image(pt, fzl[n], 0, 0)
                paste_image(pt, _fire, 385, 710, 253, 128)

            if n >= 22:
                if n - 22 <= len(fzr) - 1:
                    paste_image(pt, fzr[n - 22], 0, 0)
                    paste_image(pt, _fire, 1302, 710, 253, 128)

            if n >= 45:
                if n - 45 <= len(fzm) - 1:
                    paste_image(pt, fzm[n - 45], 0, 0)
                    paste_image(pt, _fire, 919, 710, 253, 128)

            if n >= 52:
                if n - 52 <= len(fzdl) - 1:
                    paste_image(pt, fzdl[n - 52], 0, 0)

            pt.copy().convert("RGB").save(f"./temp/scene_3/{n}.jpg")
            self.clear_pt()
        '''

        for n in range(max_frame):
            pt_fms.append(Image.open(f"./temp/scene_3/{n}.jpg"))

        ftimes = get_times_continuous(202, "9.01", "12.24")
        for _t in ftimes:
            d_fm[2].append(_t)

        self.clear_pt()
        return pt_fms

    def scene_4(self):
        ret_frames = []

        pt = self.pt

        im = Image.open("./src/灯笼.png")
        cdl = generate_turn_around_frames(im, 124, 26, [[1247, 84], [0, 84]],
                                          wh=(672, 996), color=(0, 0, 0), bgimg=pt)  # 12.24 - 13.20, 旋转灯笼

        rldl = generate_turn_around_frames(im, 0, 57, [[1247, 84], [250, 84]],
                                           wh=(672, 996), color=(0, 0, 0), bgimg=pt)  # 14.10 - 16.07, 右往左移

        cd2l = generate_turn_around_frames(im, 124, 30, [[0, 84], [1247, 84]],
                                           wh=(672, 996), color=(0, 0, 0), bgimg=pt)  # 16.08 - 17.08, 旋转灯笼, 左往右

        rl2dl = generate_turn_around_frames(im, 0, 27, [[1247, 84], [250, 84]],
                                            wh=(672, 996), color=(0, 0, 0), bgimg=pt)  # 17.09 - 18.06, 右往左移

        for gf in [cdl, rldl, cd2l, rl2dl]:
            for _t in gf:
                ret_frames.append(_t)

        ftimes = get_times_continuous(140, "12.24", "18.06")
        for _t in ftimes:
            d_fm[3].append(_t)

        self.clear_pt()
        return ret_frames

    def scene_5(self):
        ret_frames = []

        pt = self.pt
        im = Image.open("./src/灯笼.png")

        paste_image(pt, im, 449, 84, 672, 996)  # 18.07
        ret_frames.append(pt.copy())
        self.clear_pt()

        paste_image(pt, im, 449, 0, 870, 1289)
        ret_frames.append(pt.copy())
        self.clear_pt()

        paste_image(pt, im, 350, 0, 1046, 1550)
        ret_frames.append(pt.copy())
        self.clear_pt()

        paste_image(pt, im, 273, 0, 1235, 1830)
        ret_frames.append(pt.copy())
        self.clear_pt()

        ret_frames.append(pt.copy())  # 19.06

        paste_image(pt, im, 805, 125, 591, 876)  # 19.09
        ret_frames.append(pt.copy())
        self.clear_pt()

        paste_image(pt, im, 449, 0, 870, 1289)  # 19.29
        ret_frames.append(pt.copy())
        self.clear_pt()

        paste_image(pt, im, 350, -200, 1046, 1550)  # 20.01
        ret_frames.append(pt.copy())
        self.clear_pt()

        paste_image(pt, im, 273, -200, 1235, 1830)  # 20.19
        ret_frames.append(pt.copy())
        ret_frames.append(pt.copy())  # 21.07
        self.clear_pt()

        self.clear_pt()
        return ret_frames

    def scene_6(self):
        ret_frames = []

        pt = self.pt
        im = Image.open("./src/灯笼.png")

        move_light = generate_turn_around_frames(im, 0, 44, [[-504, -638], [1920, 1080]],
                                                 wh=(1141, 1691), color=(0, 0, 0), bgimg=pt)  # 21.08 - 22.22 灯笼移动
        ret_frames += move_light

        ftimes = get_times_continuous(44, "21.08", "22.22")
        for _t in ftimes:
            d_fm[5].append(_t)

        fzs = [1] * 10

        _n = 0

        # 22.24 - 29.05, 191f

        count = 0
        flags = {}
        last_flag_index = -1
        for n in range(191):  # 7帧一炮
            if count % 7 == 0:
                if last_flag_index <= len(fzs) - 1:
                    last_flag_index += 1
                else:
                    last_flag_index = 0
                flags[last_flag_index] = 0

            for f_index in flags:
                flags[f_index] += 1

            print(flags)

            rm_index = []
            for f_index in flags:
                t_frame = flags[f_index]
                if t_frame >= 30:
                    rm_index.append(f_index)
            for _f in rm_index:
                flags.pop(_f)

            count += 1

        ftimes = get_times_continuous(191, "22.24", "29.05")
        for _t in ftimes:
            d_fm[5].append(_t)

        self.clear_pt()

        for i in range(191):
            ret_frames.append(Image.open(f"./temp/scene_5/{i}.jpg"))

        return ret_frames

    def scene_7(self):
        ret_frames = []

        # pt = Image.new("RGBA", (1920, 1080), (255, 255, 255, 0))
        # im = Image.open("./src/cbg.png")
        # ft = Image.open("./src/fu.png")

        # rfb1 = generate_turn_around_frames(im, 13, 179, [[192, 456], [388, 264], [582, 118], [910, 56], [1172, 162],
        #                                                 [1286, 408], [1080, 624], [724, 540], [502, 438]],
        #                                   wh=(378, 378), color=(0, 0, 0), bgimg=pt)  # 29.06 - 35.05
        # rf1 = generate_turn_around_frames(ft, 63, 179, [[192, 456], [388, 264], [582, 118], [910, 56], [1172, 162],
        #                                                [1286, 408], [1080, 624], [724, 540], [502, 438]],
        #                                  wh=(378, 378), color=(0, 0, 0), bgimg=pt)  # 29.06 - 35.05

        t_len = len(range(179))
        for i in range(t_len):
            # pt2 = paste_image_return(rfb1[i], rf1[i], 0, 0, fill=(255, 255, 255))
            # pt2 = paste_image_return(pt2, rfb1[i], 0, 0, fill=(255, 255, 255), x_mirror=True)
            # pt2 = paste_image_return(pt2, rf1[i], 0, 0, fill=(255, 255, 255), x_mirror=True)
            # pt2 = paste_image_return(pt2, rfb1[i], 0, 0, fill=(255, 255, 255), y_mirror=True)
            # pt2 = paste_image_return(pt2, rf1[i], 0, 0, fill=(255, 255, 255), y_mirror=True)
            # pt2 = paste_image_return(pt2, rfb1[i], 0, 0, fill=(255, 255, 255), x_mirror=True, y_mirror=True)
            # pt2 = paste_image_return(pt2, rf1[i], 0, 0, fill=(255, 255, 255), x_mirror=True, y_mirror=True)
            # pt2.convert("RGB").save(f"./temp/scene_6/{i}.jpg", quality=90)
            print("scene_7", f"{i}/{t_len}")

        for i in range(t_len):
            try:
                ret_frames.append(Image.open(f"./temp/scene_6/{i}.jpg"))
            except FileNotFoundError:
                pass

        ftimes = get_times_continuous(179, "29.06", "35.05")
        for _t in ftimes:
            d_fm[6].append(_t)

        return ret_frames


fms = BaiLanFrames()

r_fms: List[Callable] = [
    fms.scene_1,
    fms.scene_2,
    fms.scene_3,
    fms.scene_4,
    fms.scene_5,
    fms.scene_6,
    fms.scene_7
]

d_fm = [  # 每帧延时
    [0.08, 1.05, 1.26, 2.08, 2.09],
    [2.09, 2.21, 3.03, 3.19, 4.02],
    [], [],
    [18.07, 18.08, 18.11, 18.18, 19.06, 19.09, 19.29, 20.01, 20.19, 21.07], [], []
]
