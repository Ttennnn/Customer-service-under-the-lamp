/* Shared back-office navigation (ch09).
 *
 * Every admin/review/observability/kb/rag-eval page loads this file and
 * renders the same five links through mountAdminNav().  The nav styling is
 * injected once and falls back to neutral colours on pages that do not define
 * the admin CSS variables, so no page needs a new stylesheet dependency.
 */
(function () {
  "use strict";

  var ITEMS = [
    { href: "/admin", label: "后台首页" },
    { href: "/review", label: "待审队列" },
    { href: "/observability", label: "观测总览" },
    { href: "/kb", label: "知识录入" },
    { href: "/rag-eval", label: "RAG 评估" }
  ];

  var STYLE_ID = "admin-nav-style";
  var STYLE_TEXT = [
    ".admin-nav{display:flex;flex-wrap:wrap;gap:8px;align-items:center;",
    "max-width:1120px;margin:0 auto 14px;padding:8px 10px;",
    "border:1px solid var(--border,#d9dee6);border-radius:12px;",
    "background:var(--panel,var(--surface,#ffffff));",
    "color:var(--text,#1d2430);",
    "font:13px/1.5 system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}",
    ".admin-nav a{padding:5px 11px;border-radius:8px;border:1px solid transparent;",
    "color:inherit;text-decoration:none;opacity:.78;transition:opacity .15s,border-color .15s,color .15s}",
    ".admin-nav a:hover{opacity:1;border-color:var(--border,#d9dee6)}",
    ".admin-nav a.active{opacity:1;border-color:var(--accent,#2563eb);",
    "color:var(--accent,#2563eb);font-weight:600}"
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

    var path = currentPath();
    ITEMS.forEach(function (item) {
      var link = document.createElement("a");
      link.href = item.href;
      link.textContent = item.label;
      if (path === item.href) {
        link.className = "active";
        link.setAttribute("aria-current", "page");
      }
      mount.append(link);
    });
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
