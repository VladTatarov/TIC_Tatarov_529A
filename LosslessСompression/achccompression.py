import ast
import collections
import math
import matplotlib.pyplot as plt

open("results_AC_CH.txt", "w")
res = []


def main():
    with open("sequence.txt", "r") as f:
        orig_seqs = ast.literal_eval(f.read())
        orig_seqs = [s.strip("[]").strip("'") for s in orig_seqs]

    for s in orig_seqs:
        s = s[:10]
        n = len(s)
        u = set(s)
        m = len(u)
        c = collections.Counter(s)
        p = {x: c[x] / 10 for x in c}
        e = -sum(v * math.log2(v) for v in p.values())

        ed, es = enc(u, p, m, s)
        bps = len(es) / n
        ds = dec(ed, n)

        with open("results_AC_CH.txt", "a") as f:
            sl = "/-/" * 20
            txt = f"Orig sequence: {s}\nEncoded data: {ed}\nEncoded seq: {es}\nBPS: {bps}\nDecoded: {ds}"
            f.write(f"\n{sl}\n")
            f.write(txt)

        ed_hc, es_hc = enc_hc(u, p, s)
        bps_hc = len(es_hc) / n
        ds_hc = dec_hc(ed_hc)

        with open("results_AC_CH.txt", "a") as f:
            sl = "/-/" * 20
            txt = f"\nEncoded data HC: {ed_hc}\nEncoded seq HC: {es_hc}\nBPS HC: {bps_hc}\nDecoded HC: {ds_hc}\n"
            f.write(txt)
            f.write(f"\n{sl}\n")
        res.append([round(e, 2), bps, bps_hc])
    fig, ax = plt.subplots(figsize=(14 / 1.54, 14 / 1.54))
    headers = ['Ентропія', 'BPS AC', 'BPS HC']
    row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4',
           'Послідовність 5', 'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
    ax.axis('off')
    tbl = ax.table(cellText=res, colLabels=headers, rowLabels=row,
                   loc='center', cellLoc='center')
    tbl.set_fontsize(14)
    tbl.scale(0.8, 2)
    plt.savefig("Результати стиснення методами AC та CH", dpi=600)


def fl_bin(p, s):
    b = ""
    for _ in range(s):
        p *= 2
        if p > 1:
            b += str(1)
            d = int(p)
            p -= d
        elif p < 1:
            b += str(0)
        elif p == 1:
            b += str(1)
    return b


def enc(u, p, m, s):
    a = list(u)
    pr = [p[x] for x in a]
    unity = []
    pr_range = 0.0
    for _ in range(m):
        i = pr_range
        pr_range += pr[_]
        unity.append([a[_], i, pr_range])

    for _ in range(len(s) - 1):
        for __ in range(len(unity)):
            if s[_] == unity[__][0]:
                pl = unity[__][1]
                ph = unity[__][2]
                d = ph - pl
                for ___ in range(len(unity)):
                    unity[___][1] = pl
                    unity[___][2] = pr[___] * d + pl
                    pl = unity[___][2]
                break
    l = 0
    h = 0
    for _ in range(len(unity)):
        if unity[_][0] == s[-1]:
            l = unity[_][1]
            h = unity[_][2]
    pnt = (l + h) / 2
    s_cod = math.ceil(math.log((1 / (h - l)), 2) + 1)
    b_c = fl_bin(pnt, s_cod)
    return [pnt, m, a, pr], b_c


def dec(encoded_sequence, length_seq):
    ed = encoded_sequence
    pnt = ed[0]
    m = ed[1]
    a = ed[2]
    pr = ed[3]

    unity = []
    pr_range = 0.0
    for _ in range(m):
        l = pr_range
        pr_range += pr[_]
        u = pr_range
        unity.append([a[_], l, u])
    ds = ""
    for _ in range(length_seq):
        for __ in range(len(unity)):
            if unity[__][1] < pnt < unity[__][2]:
                pl = unity[__][1]
                ph = unity[__][2]
                d = ph - pl
                ds += unity[__][0]
                for ___ in range(len(unity)):
                    unity[___][1] = pl
                    unity[___][2] = pr[___] * d + pl
                    pl = unity[___][2]
                break
    return ds


def enc_hc(u, p, s):
    a = list(u)
    pr = [p[x] for x in a]
    final = []
    for _ in range(len(a)):
        final.append([a[_], pr[_]])
    final.sort(key=lambda x: x[1])

    tree = []

    if 1 in pr and len(set(pr)) == 1:
        sc = []
        for _ in range(len(a)):
            c = "1" * _ + "0"
            sc.append([a[_], c])
        encd = "".join([sc[a.index(c)][1] for c in s])
        with open("results_AC_CH.txt", "a") as f:
            f.write(f"\nHuffman coding\n")
            f.write(f"alphabet|symbol codes\n")
        for _ in range(len(sc)):
            with open("results_AC_CH.txt", "a") as f:
                f.write(f"{sc[_][0]} {sc[_][1]}\n")
    else:
        for _ in range(len(final) - 1):
            _ = 0
            left = final[_]
            final.pop(_)
            right = final[_]
            final.pop(_)
            tot = left[1] + right[1]
            tree.append([left[0], right[0]])
            final.append([left[0] + right[0], tot])
            final.sort(key=lambda x: x[1])
        sc = []
        tree.reverse()
        a.sort()
        for _ in range(len(a)):
            c = ""
            for __ in range(len(tree)):
                if a[_] in tree[__][0]:
                    c += '0'
                    if a[_] == tree[__][0]:
                        break
                else:
                    c += '1'
                    if a[_] == tree[__][1]:
                        break
            sc.append([a[_], c])
        encd = ""
        for c in s:
            encd += [sc[_][1] for _ in range(len(a)) if sc[_][0] == c][0]
        with open("results_AC_CH.txt", "a") as f:
            f.write(f"\nHuffman coding\n")
            f.write(f"alphabet|symbol codes\n")
        for _ in range(len(sc)):
            with open("results_AC_CH.txt", "a") as f:
                f.write(f"{sc[_][0]} {sc[_][1]}\n")
    return [encd, sc], encd


def dec_hc(encoded_sequence):
    e = list(encoded_sequence[0])
    sc = encoded_sequence[1]
    cnt = 0
    flag = 0
    seq = ""
    for _ in range(len(e)):
        for __ in range(len(sc)):
            if e[_] == sc[__][1]:
                seq += str(sc[__][0])
                flag = 1
        if flag == 1:
            flag = 0
        else:
            cnt += 1
            if cnt == len(e):
                break
            else:
                e.insert(_ + 1, str(e[_] + e[_ + 1]))
                e.pop(_ + 2)
    return seq


if __name__ == "__main__":
    main()
