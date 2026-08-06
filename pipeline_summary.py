# -*- coding: utf-8 -*-
"""daily_pipeline 완료 알림용 요약 텍스트 출력 (forward 추적 현황)."""
import sys
import psycopg2

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    c = psycopg2.connect(host="localhost", port=5432, dbname="equity", user="postgres")
    cur = c.cursor()

    cur.execute("""SELECT snapshot_date, total_equity, cum_return_pct,
                          daily_return_pct, n_open_positions,
                          n_entries_today, n_exits_today
                   FROM forward_capital ORDER BY snapshot_date DESC LIMIT 1""")
    r = cur.fetchone()

    cur.execute("SELECT max(signal_date) FROM forward_signals")
    sd = cur.fetchone()[0]
    picks = []
    if sd:
        cur.execute("SELECT ticker FROM forward_signals WHERE signal_date=%s ORDER BY rank", (sd,))
        picks = [x[0] for x in cur.fetchall()]

    cur.execute("SELECT max(trade_date)::text FROM market_data")
    md = cur.fetchone()[0]
    c.close()

    lines = []
    if r:
        lines.append(f"기준일: {r[0]}   (데이터 {md})")
        lines.append(f"자산: {float(r[1]):,.0f}원")
        lines.append(f"수익률: 일간 {float(r[3] or 0):+.2f}% / 누적 {float(r[2] or 0):+.2f}%")
        lines.append(f"보유: {r[4]}종목   오늘 진입 {r[5]} / 청산 {r[6]}")
    else:
        lines.append("forward_capital 비어있음 (아직 사이클 미실행)")
    lines.append(f"최신 추천일: {sd if sd else '없음'}")
    lines.append(f"추천종목: {', '.join(picks) if picks else '없음'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
