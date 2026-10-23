"""Headless-Chrome E2E for IdeaGen against the ISOLATED stack (frontend :5180 -> backend :8765 on a DB copy).

Workflows: A (legacy local project registration + modal create), B (search/add/navigate/reload),
C (duplicate), D (isolation). Real OpenAlex is used; nothing here touches the demo DB.
"""
import asyncio, base64, json, subprocess, sys, time, urllib.request
import websockets

FE = "http://127.0.0.1:5180"
BE = "http://127.0.0.1:8765"
SHOTS = "/tmp/ideagen_audit/shots"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CYBER = {
    "id": "proj_1761000000000_zday1",
    "name": "Adaptive AI for Zero-Day Attack Detection",
    "topic": "Adaptive machine learning for detecting zero-day network attacks",
    "description": "",
    "createdAt": "2026-10-22T08:00:00.000Z",
    "updatedAt": "2026-10-22T08:00:00.000Z",
    "status": "ACTIVE",
    "temporalCutoff": "2024-12-31",
    "isDefaultBenchmark": False,
}
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def api(path, method="GET", body=None):
    req = urllib.request.Request(BE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


class Page:
    def __init__(self, ws):
        self.ws, self.i = ws, 0

    async def cmd(self, method, params=None):
        self.i += 1
        await self.ws.send(json.dumps({"id": self.i, "method": method, "params": params or {}}))
        while True:
            r = json.loads(await self.ws.recv())
            if r.get("id") == self.i:
                return r

    async def js(self, expr):
        r = await self.cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True})
        return r["result"]["result"].get("value")

    async def goto(self, url, wait=2.5):
        await self.cmd("Page.navigate", {"url": url})
        await asyncio.sleep(wait)

    async def wait_for(self, expr, timeout=40):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if await self.js(expr):
                return True
            await asyncio.sleep(0.5)
        return False

    async def shot(self, name, width=1280, height=1600):
        await self.cmd("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 600})
        await asyncio.sleep(0.6)
        r = await self.cmd("Page.captureScreenshot", {"format": "png"})
        open(f"{SHOTS}/{name}.png", "wb").write(base64.b64decode(r["result"]["data"]))
        await self.cmd("Emulation.setDeviceMetricsOverride", {"width": 1280, "height": 900, "deviceScaleFactor": 1, "mobile": False})

    async def overflow(self, width):
        await self.cmd("Emulation.setDeviceMetricsOverride", {"width": width, "height": 900, "deviceScaleFactor": 1, "mobile": width < 600})
        await asyncio.sleep(0.8)
        sw = await self.js("document.documentElement.scrollWidth")
        await self.cmd("Emulation.setDeviceMetricsOverride", {"width": 1280, "height": 900, "deviceScaleFactor": 1, "mobile": False})
        return sw

    async def set_value(self, selector, value):
        return await self.js(f"""(() => {{
          const el = document.querySelector({json.dumps(selector)}); if (!el) return false;
          const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : (el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype);
          Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, {json.dumps(value)});
          el.dispatchEvent(new Event('input', {{bubbles: true}})); el.dispatchEvent(new Event('change', {{bubbles: true}}));
          return true; }})()""")

    async def click_text(self, text, selector="button"):
        return await self.js(f"""(() => {{
          const b = [...document.querySelectorAll({json.dumps(selector)})].find(e => e.textContent.trim().includes({json.dumps(text)}) && !e.disabled);
          if (!b) return false; b.click(); return true; }})()""")

    async def text(self):
        return await self.js("document.body.innerText")


async def main(ws_url):
    async with websockets.connect(ws_url, max_size=None) as ws:
        p = Page(ws)
        await p.cmd("Page.enable")
        await p.cmd("Emulation.setDeviceMetricsOverride", {"width": 1280, "height": 900, "deviceScaleFactor": 1, "mobile": False})

        # ── Workflow A: a project that exists only in this browser's localStorage (pre-fix state)
        await p.goto(FE + "/#/dashboard", 1.5)
        seed = json.dumps([CYBER])
        await p.js(f"localStorage.setItem('ideagen_projects_v1', {json.dumps(seed)}); localStorage.setItem('ideagen_active_project_id', '{CYBER['id']}'); sessionStorage.clear(); true")
        status, before = api("/api/v1/projects")
        check("A0 cyber project absent on server before sync", CYBER["id"] not in [x["id"] for x in before])
        await p.cmd("Page.reload", {"ignoreCache": True})
        await asyncio.sleep(4)
        status, after = api("/api/v1/projects")
        srv = {x["id"]: x for x in after}
        check("A1 legacy local project registered on server with same id", CYBER["id"] in srv)
        check("A2 registered cutoff/title preserved", srv.get(CYBER["id"], {}).get("temporal_cutoff") == "2024-12-31"
              and srv.get(CYBER["id"], {}).get("title") == CYBER["name"])
        page_text = await p.text()
        check("A3 dashboard shows API Connected", "API Connected" in page_text)
        check("A4 dashboard has no fabricated 'Stage 5 Evolution Complete'", "Stage 5 Evolution Complete" not in page_text)
        await p.shot("e2e_dashboard_1280")
        check("A5 dashboard no horizontal overflow @390", (await p.overflow(390)) <= 390)
        await p.shot("e2e_dashboard_390", 390, 2400)

        # ── Workflow B: real OpenAlex search, add to project, navigate away/back, reload
        await p.goto(FE + f"/#/projects/{CYBER['id']}/papers?tab=search", 2.5)
        await p.set_value("input[type=search]", "zero-day attack detection machine learning")
        await p.js("document.querySelector('form[role=search]').requestSubmit(); true")
        found = await p.wait_for("[...document.querySelectorAll('button')].filter(b => b.textContent.includes('Add to Project Corpus')).length > 0", 45)
        check("B1 live OpenAlex search returned results in UI", found)
        await p.shot("e2e_search_results_1280")
        check("B1b search page no horizontal overflow @390", (await p.overflow(390)) <= 390)
        await p.shot("e2e_search_390", 390, 2400)
        title = await p.js("(() => { const b = [...document.querySelectorAll('button')].filter(b => b.textContent.includes('Add to Project Corpus'))[0]; return b ? b.getAttribute('aria-label').slice(5, -19) : null })()")
        await p.js("[...document.querySelectorAll('button')].filter(b => b.textContent.includes('Add to Project Corpus'))[0].click(); true")
        added = await p.wait_for("/Added to Project|Already in Project|Ingestion Failed|failed/i.test(document.body.innerText)", 90)
        txt = await p.text()
        check("B2 add result rendered", added and ("Added to Project" in txt), (txt[txt.find("Added to Project") - 40: txt.find("Added to Project") + 60] if "Added to Project" in txt else "no success badge"))
        await p.shot("e2e_search_added_1280")
        status, cyber_papers = api(f"/api/v1/papers?project_id={CYBER['id']}")
        check("B3 paper persisted & associated with cyber project (API)", len(cyber_papers) == 1 and cyber_papers[0]["title"] == title, f"{len(cyber_papers)} paper(s): {[x['title'][:50] for x in cyber_papers]}")
        await p.goto(FE + "/#/dashboard", 2)
        await p.goto(FE + f"/#/projects/{CYBER['id']}/papers?tab=corpus", 3)
        check("B4 Corpus Inspector shows paper after navigating back", title and title[:40] in await p.text())
        await p.shot("e2e_corpus_1280")
        await p.cmd("Page.reload", {"ignoreCache": True})
        await asyncio.sleep(4)
        check("B5 Corpus Inspector shows paper after full reload", title and title[:40] in await p.text())
        await p.goto(FE + f"/#/projects/{CYBER['id']}/papers?tab=search", 3)
        check("B6 search results restored deliberately from session state", title and title[:40] in await p.text())

        # ── Workflow C: duplicate add (UI disables the button; API is idempotent)
        oid = cyber_papers[0]["source_id"] if cyber_papers else None
        status, dup = api("/api/v1/papers/ingest/openalex", "POST", {"openalex_id": oid, "project_id": CYBER["id"]})
        check("C1 duplicate add returns SKIPPED_DUPLICATE + already_in_project", status == 200 and dup.get("status") == "SKIPPED_DUPLICATE" and dup.get("already_in_project") is True, json.dumps({k: dup.get(k) for k in ("status", "already_in_project", "project_associated")}))
        status, again = api(f"/api/v1/papers?project_id={CYBER['id']}")
        check("C2 still exactly one corpus paper", len(again) == 1)

        # ── Workflow D: isolation from the RAG benchmark
        status, rag = api("/api/v1/papers?project_id=rag-benchmark&limit=100")
        check("D1 rag-benchmark corpus unchanged (11) and does not contain the cyber paper", len(rag) == 11 and all(x["id"] not in {c["id"] for c in cyber_papers} for x in rag))
        status, rag_stats = api("/api/v1/stats?project_id=rag-benchmark")
        check("D2 rag-benchmark stats unchanged", rag_stats["total_papers"] == 11 and rag_stats["total_gaps"] == 2, json.dumps({k: rag_stats[k] for k in ("total_papers", "total_gaps", "total_opportunities")}))

        # ── Workflow A (cont.): create a project through the modal; it must exist on the server
        await p.goto(FE + "/#/dashboard", 3)
        await p.click_text("New Project")
        await asyncio.sleep(0.8)
        await p.shot("e2e_create_modal_1280", 1280, 1100)
        await p.set_value("#proj-name", "E2E Isolation Probe")
        await p.set_value("#proj-topic", "Temporary project created by the audit browser test")
        await p.set_value("#proj-cutoff", "2022-06-30")
        await p.js("document.querySelector('#proj-name').form.requestSubmit(); true")
        await asyncio.sleep(3)
        status, projs = api("/api/v1/projects")
        probe = [x for x in projs if x["title"] == "E2E Isolation Probe"]
        before_ids = {x["id"] for x in before}
        probe = [x for x in probe if x["id"] not in before_ids and x["created_at"]]
        probe = sorted(probe, key=lambda x: x["created_at"])[-1:]
        check("A6 modal-created project persisted on server", len(probe) == 1 and probe[0]["temporal_cutoff"] == "2022-06-30")
        if probe:
            status, probe_papers = api(f"/api/v1/papers?project_id={probe[0]['id']}")
            check("D3 new project corpus is empty (no leakage)", probe_papers == [])

        # ── Overview / settings / gaps / history pages at laptop + narrow widths
        for route, name in [(f"/#/projects/{CYBER['id']}/overview", "overview_cyber"), ("/#/projects/rag-benchmark/overview", "overview_rag"),
                            ("/#/settings", "settings"), ("/#/projects/rag-benchmark/gaps", "gaps_rag"),
                            ("/#/projects/rag-benchmark/opportunities", "opps_rag"), ("/#/projects/rag-benchmark/history", "history_rag"),
                            ("/#/projects/rag-benchmark/evolution", "evolution_rag")]:
            await p.goto(FE + route, 3.5)
            t = await p.text()
            await p.shot(f"e2e_{name}_1280")
            sw = await p.overflow(390)
            check(f"UI {name}: renders, no overflow @390 (scrollWidth={sw})", sw <= 390 and "Something went wrong" not in t)
            if name == "overview_rag":
                st = api("/api/v1/stats?project_id=rag-benchmark")[1]
                check("UI overview_rag shows API-derived passages and no hardcoded health", f"from {st['total_passages']} stored passages" in t and "Connected & Verified" not in t)
            if name == "settings":
                check("UI settings shows live model availability", "Installed" in t and "(Online)" not in t)
            if name == "history_rag":
                check("UI history has no 'undefined' stage count", "undefined" not in t)
    return results


if __name__ == "__main__":
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--remote-debugging-port=9334",
                             "--user-data-dir=/tmp/ideagen_audit/chrome_e2e", "--window-size=1280,900", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9334/json"))
                break
            except Exception:
                time.sleep(0.25)
        ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
        asyncio.run(main(ws_url))
    finally:
        proc.terminate()
    failed = [r for r in results if not r[1]]
    print(f"\nSUMMARY: {len(results) - len(failed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
