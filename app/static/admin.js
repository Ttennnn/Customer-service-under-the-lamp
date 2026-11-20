/* Shared back-office navigation (ch09, restyled with the chat theme).
 *
 * Every admin/review/observability/topics/acceptance/kb/rag-eval page loads
 * this file and renders the same seven links through mountAdminNav(). The
 * styling is injected once and falls back to neutral colours on pages that do
 * not define the shared CSS variables.
 */
(function () {
  "use strict";

  var ITEMS = [
    { href: "/admin", label: "后台首页", match: function (p) { return p === "/admin"; } },
    { href: "/review", label: "待审队列", match: function (p) { return p === "/review"; } },
    { href: "/observability", label: "观测总览", match: function (p) { return p === "/observability"; } },
    { href: "/topics", label: "主题分布", match: function (p) { return p === "/topics" || p.indexOf("/topics/") === 0; } },
    { href: "/acceptance", label: "验收", match: function (p) { return p === "/acceptance" || p.indexOf("/acceptance/") === 0; } },
    { href: "/kb", label: "知识录入", match: function (p) { return p === "/kb"; } },
    { href: "/rag-eval", label: "RAG 评估", match: function (p) { return p === "/rag-eval"; } }
  ];

  var BRAND_SVG =
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M12 3v5"/><path d="M5.5 15a6.5 6.5 0 0 1 13 0Z"/>' +
    '<circle cx="12" cy="17.5" r="1.25" fill="currentColor" stroke="none"/></svg>';

  var STYLE_ID = "admin-nav-style";
  var STYLE_TEXT = [
    ".admin-nav{display:flex;align-items:center;gap:14px;max-width:1120px;margin:0 auto;",
    "padding:10px 0;border-bottom:1px solid var(--border,rgba(240,233,223,.08));",
    "color:var(--text,#1d2430);font:13px/20px var(--sans,system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif)}",
    ".admin-nav .brand{flex:0 0 auto;display:inline-flex;align-items:center;gap:8px;",
    "color:var(--text,#1d2430);text-decoration:none;font-size:14px;font-weight:500;letter-spacing:0}",
    ".admin-nav .brand svg{color:var(--accent,#2563eb)}",
    ".admin-nav .links{display:flex;align-items:center;gap:2px;min-width:0;overflow-x:auto;scrollbar-width:none}",
    ".admin-nav .links::-webkit-scrollbar{display:none}",
    ".admin-nav .links a{flex:0 0 auto;display:inline-flex;align-items:center;height:32px;padding:0 9px;",
    "border-radius:6px;color:var(--dim,#5c6675);text-decoration:none;font-size:12.5px;letter-spacing:0;",
    "transition:background-color .15s,color .15s}",
    ".admin-nav .links a:hover{background:var(--raised,#f4f4f5);color:var(--text,#1d2430)}",
    ".admin-nav .links a.active{background:var(--raised,#f4f4f5);color:var(--text,#1d2430);",
    "box-shadow:inset 0 1px 0 rgba(255,255,255,.06)}",
    "@media (max-width:640px){.admin-nav{padding:8px 0;gap:10px}.admin-nav .brand span{display:none}}"
  ].join("");

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLE_TEXT;
    document.head.appendChild(style);
  }

  function currentPath() {
    var path = window.location.pathname || "/";
    if (path.length > 1) path = path.replace(/\/+$/, "");
    return path || "/";
  }

  function mountAdminNav(target) {
    if (typeof target === "string") target = document.getElementById(target);
    var mount = target || document.getElementById("admin-nav");
    if (!mount) {
      mount = document.createElement("nav");
      mount.id = "admin-nav";
      if (document.body) {
        document.body.insertBefore(mount, document.body.firstChild);
      } else {
        return null;
      }
    }

    ensureStyle();
    mount.className = "admin-nav";
    mount.setAttribute("aria-label", "后台导航");
    mount.textContent = "";

    var brand = document.createElement("a");
    brand.className = "brand";
    brand.href = "/admin";
    brand.innerHTML = BRAND_SVG + "<span>灯下后台</span>";
    mount.appendChild(brand);

    var links = document.createElement("div");
    links.className = "links";
    var path = currentPath();
    ITEMS.forEach(function (item) {
      var link = document.createElement("a");
      link.href = item.href;
      link.textContent = item.label;
      if (item.match(path)) {
        link.className = "active";
        link.setAttribute("aria-current", "page");
      }
      links.appendChild(link);
    });
    mount.appendChild(links);
    return mount;
  }

  window.mountAdminNav = mountAdminNav;

  // Pages that only load the script (kb, rag-eval) still get the nav; pages
  // with dynamic content may call mountAdminNav() again after rendering.
  function autoMount() {
    mountAdminNav();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", autoMount);
  } else {
    autoMount();
  }
})();
