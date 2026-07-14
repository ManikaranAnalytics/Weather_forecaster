# MAMS — Design System & Layout Reference

> **LLM Context File** — Feed this to any LLM working on a new project to replicate the exact visual language, layout structure, and component patterns used in MAMS.
> Last updated: 2026-06-01

---

## 1. Overview

MAMS uses a **premium enterprise dashboard** aesthetic. The design philosophy is:

- **Dark-first**: Dark mode is default. Light mode is a clean warm-white variant.
- **Glassmorphism**: Translucent cards and sidebars with `backdrop-blur`.
- **Micro-animations**: Every interactive element has hover/active transforms.
- **Uppercase labels**: Section labels, badges, and captions are always uppercase with wide tracking.
- **No emojis or decorative pictographs** in UI text or code.

---

## 2. Tech Stack for UI

| Layer      | Technology                  | Notes                                      |
|------------|-----------------------------|--------------------------------------------|
| Framework  | Next.js 16 (App Router)     | Server + Client components                 |
| Styling    | **Tailwind CSS v4**         | `@import "tailwindcss"` in globals.css     |
| Icons      | **Lucide React** v1.7.0     | Only icon library used                     |
| Fonts      | **Plus Jakarta Sans** (body) | Variable: `--font-sans`                   |
|            | **JetBrains Mono** (code)   | Variable: `--font-mono`                   |
| Animation  | Tailwind `animate-in` + custom `@keyframes fadeIn` | |
| Charts     | Highcharts + highcharts-react-official | Admin reports only     |

---

## 3. Color Palette & CSS Variables

All colors are defined as CSS variables and mapped into Tailwind via `@theme inline`.

### Light Theme (`:root`)

```css
--background:         #fdfcf9;   /* Warm off-white page bg */
--foreground:         #1a1a1a;   /* Near-black text */
--primary:            #185FA5;   /* Deep blue — primary actions */
--primary-foreground: #ffffff;
--secondary:          #085041;   /* Deep green — secondary actions */
--secondary-foreground: #ffffff;
--accent:             #ef9f27;   /* Amber/gold — highlights, warnings */
--accent-foreground:  #1a1a1a;
--muted:              #f3f2ee;   /* Light warm grey — muted surfaces */
--muted-foreground:   #6c6a62;   /* Greyed out text */
--border:             #e6e4dc;   /* Warm grey borders */
--card:               #ffffff;   /* Card backgrounds */
--card-foreground:    #1a1a1a;
--glass-bg:           rgba(255, 255, 255, 0.7);
--glass-border:       rgba(156, 154, 146, 0.2);
```

### Dark Theme (`.dark` / `[data-theme="dark"]`)

```css
--background:         #0f1115;   /* Very dark near-black */
--foreground:         #e2e8f0;   /* Slate-200 text */
--primary:            #1e81cf;   /* Bright blue */
--primary-foreground: #ffffff;
--secondary:          #10b981;   /* Emerald-500 green */
--secondary-foreground: #000000;
--accent:             #f59e0b;   /* Amber-500 */
--accent-foreground:  #000000;
--muted:              #1e293b;   /* Slate-800 */
--muted-foreground:   #94a3b8;   /* Slate-400 */
--border:             #334155;   /* Slate-700 */
--card:               #1e293b;   /* Slate-800 */
--card-foreground:    #e2e8f0;
--glass-bg:           rgba(15, 17, 21, 0.8);
--glass-border:       rgba(51, 65, 85, 0.4);
```

### Tailwind Theme Mapping

```css
@theme inline {
  --color-background:        var(--background);
  --color-foreground:        var(--foreground);
  --color-primary:           var(--primary);
  --color-primary-foreground: var(--primary-foreground);
  --color-secondary:         var(--secondary);
  --color-secondary-foreground: var(--secondary-foreground);
  --color-accent:            var(--accent);
  --color-accent-foreground: var(--accent-foreground);
  --color-muted:             var(--muted);
  --color-muted-foreground:  var(--muted-foreground);
  --color-border:            var(--border);
  --color-card:              var(--card);
  --color-card-foreground:   var(--card-foreground);
  --font-sans: var(--font-sans), ui-sans-serif, system-ui, sans-serif;
  --font-mono: var(--font-mono), ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
```

---

## 4. Typography System

| Use Case              | Classes                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| Page title (header)   | `text-xl font-semibold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent` |
| Page subtitle         | `text-[10px] font-black uppercase tracking-[0.24em] text-muted-foreground/80` |
| Section heading       | `text-sm font-black uppercase tracking-widest`                         |
| Group label (sidebar) | `text-[9px] font-semibold text-muted-foreground/60 uppercase tracking-widest` |
| Card title            | `text-sm font-medium text-muted-foreground uppercase tracking-widest`  |
| Card value            | `text-3xl font-bold tracking-tight`                                    |
| Body text             | `text-sm font-medium` or `text-xs font-bold`                           |
| Micro-label / badge   | `text-[7px] font-black uppercase tracking-widest`                      |
| Caption               | `text-[10px] font-black uppercase tracking-tighter`                    |
| Code / mono           | `font-mono text-xs`                                                     |

**Key rule:** Labels are **ALWAYS uppercase**. Use `font-black` for emphasis labels and `tracking-widest` for spaced-out caps.

---

## 5. Page Layout Architecture

### 5.1 Full Page Shell

```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR (fixed, left)          MAIN CONTENT AREA           │
│  ┌───────────────┐              ┌────────────────────────┐   │
│  │  Logo / Brand │              │  HEADER (sticky top)   │   │
│  │               │              │  Title | Search |      │   │
│  │  NAV LINKS    │              │  Scope | Theme | User  │   │
│  │  (grouped)    │              ├────────────────────────┤   │
│  │               │              │                        │   │
│  │               │              │  PAGE CONTENT          │   │
│  │               │              │  p-4 md:p-6 pb-20      │   │
│  │               │              │                        │   │
│  ├───────────────┤              │                        │   │
│  │  User Card    │              │                        │   │
│  │  Log Out btn  │              │                        │   │
│  └───────────────┘              └────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Layout Measurements

| Element            | Expanded             | Collapsed            |
|--------------------|----------------------|----------------------|
| Sidebar width      | `w-64` (256px)       | `w-20` (80px)        |
| Main left padding  | `pl-64`              | `pl-20`              |
| Header height      | `min-h-16` (64px)    | Same                 |
| Page content padding | `p-4 md:p-6 pb-20` | Same                 |
| Sidebar z-index    | `z-50`               | `z-50`               |
| Header z-index     | `z-30`               | `z-30`               |

### 5.3 Login Page Exception

The `SidebarWrapper` detects `/login` route and renders children directly — **no sidebar, no header**.

---

## 6. Sidebar Component (`Sidebar.tsx`)

### Structure

```
<aside> [fixed, full-height, left-0, top-0, z-50]
  ├── Toggle button [-right-4, absolute, rounded-full, bg-primary]
  └── Scrollable content div [flex-col, overflow-y-auto]
       ├── Logo area
       │    ├── <img> — theme-aware logo (dark: MPLWhite.png, light: mrllogo.png)
       │    └── "Asset Management System" caption [only when expanded]
       ├── <nav> [px-3, pb-6, space-y-4]
       │    └── SidebarGroup[] — rendered per module
       │         ├── Group label [text-[9px], uppercase, tracking-widest]
       │         └── NavLink[]
       │              ├── Icon [w-4 h-4]
       │              ├── Link text [flex-1, truncate] — hidden when collapsed
       │              ├── Notification badge [red, animate-pulse] — conditional
       │              └── Active indicator dot [animate-pulse] — conditional
       └── User footer [border-t]
            ├── User card [bg-muted, rounded-lg, p-2]
            │    ├── Avatar [w-8 h-8, rounded-full, bg-primary, initials]
            │    ├── Name [text-sm, font-semibold]
            │    └── Role [text-[10px], uppercase, font-black]
            └── Log Out button [text-xs, uppercase, hover:text-red-500]
```

### Key Classes

```tsx
// Sidebar container
"h-screen border-r border-border bg-card flex flex-col glass fixed left-0 top-0 transition-all duration-300 ease-in-out z-50 overflow-visible"

// Collapse toggle button
"absolute top-6 -right-4 w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center shadow-[0_4px_12px_rgba(0,0,0,0.15)] hover:scale-110 active:scale-95 transition-all z-[100] border-2 border-background"

// Active nav link
"bg-primary text-primary-foreground shadow-sm"

// Inactive nav link
"text-muted-foreground hover:text-foreground hover:bg-muted"

// Nav link base
"flex items-center gap-3 px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200 group relative overflow-hidden"

// Notification badge
"flex items-center justify-center text-[10px] font-black rounded-full bg-red-500 text-white ring-2 ring-red-500/30 animate-pulse w-5 h-5"
```

### Navigation Groups & Links

| Group       | Roles Allowed               | Links                                                                      |
|-------------|-----------------------------|----------------------------------------------------------------------------|
| Overview    | hr, it, admin, readonly     | Dashboard (`/`), Seats Registry (`/seats`)                                 |
| HR Module   | hr, admin, readonly         | Upcoming Joinings, Onboarding, Employees, Exits, Attendance                |
| IT Module   | it, admin, readonly         | Assets, Provisioning, Assignments, Email Accounts, Accessories             |
| Management  | admin only                  | Master Data, Reports, Audit Log, Users                                     |

---

## 7. Header Component (`UserHeader.tsx`)

### Structure

```
<header> [sticky top-0, z-30, min-h-16, border-b, bg-card/60, backdrop-blur-md, px-6]
  ├── Left: Page Title Block
  │    ├── Title [gradient text: from-primary to-secondary, text-xl, font-semibold]
  │    └── Subtitle [text-[10px], font-black, uppercase, tracking-[0.24em], muted-foreground/80]
  └── Right: Controls (ml-auto, flex, gap-4)
       ├── OmniSearch trigger button [hidden sm:flex, bg-muted/60, rounded-xl, w-48 lg:w-64]
       │    ├── Search icon
       │    ├── "Search everything..." placeholder text
       │    └── ⌘K keyboard shortcut badge [hidden lg:flex]
       ├── ScopeSelector [location/company filter dropdown]
       ├── ThemeToggle [pill toggle: Sun/Moon icons]
       └── User profile dropdown
            ├── Trigger: Name + Role (hidden on mobile) + Avatar + ChevronDown
            └── Dropdown menu [w-64, rounded-2xl, shadow-2xl, glass]
                 ├── User info header [name, email, role badge]
                 ├── My Profile button
                 ├── Settings button
                 └── Log Out button [text-red-500]
```

### Dynamic Page Titles

The header reads the current pathname and renders a matching title + subtitle:

| Route Pattern                          | Title                    | Subtitle                                          |
|----------------------------------------|--------------------------|---------------------------------------------------|
| `/`                                    | Dashboard                | Live Asset                                        |
| `/hr/employees`                        | Employee Directory       | Workforce Identity Directory                      |
| `/hr/employees/new`                    | Employee Onboarding      | New Workforce Entry Builder                       |
| `/hr/joiners`                          | Joiner Onboarding        | List of Pending Joiners (Assets, Emails & Training)|
| `/hr/exits`                            | Exit Control             | Offboarding and Recovery Queue                    |
| `/hr/upcoming`                         | Upcoming Joinings        | List of Upcoming Joinings                         |
| `/hr/attendance`                       | Attendance Intelligence  | Monthly Workforce Attendance Overview             |
| `/hr/attendance/risk`                  | Risk Register            | Attendance Risk Scoring & Analysis                |
| `/hr/attendance/history`               | Import History           | Attendance Data Upload & Audit Trail              |
| `/hr/attendance/employee/[code]`       | Employee Attendance      | Monthly Attendance Calendar & Detail              |
| `/hr/employees/[id]`                   | Employee Profile         | Personnel Detail & Asset Allocation               |
| `/hr/employees/[id]/edit`              | Edit Employee            | Update Personnel Record                           |
| `/seats`                               | Seats Registry           | Common Workspace Occupancy & Hardware Map         |
| `/it/assets`                           | Hardware Assets          | Device Inventory Dashboard                        |
| `/it/assets/new`                       | Asset Onboarding         | New Hardware Intake Workspace                     |
| `/it/assets/[id]/edit`                 | Asset Revision           | Configuration and Lifecycle Editing               |
| `/it/assets/[id]`                      | Asset Profile            | Detailed Lifecycle and Assignment View            |
| `/it/accessories`                      | Accessory Inventory      | Peripheral Readiness Board                        |
| `/it/provisioning`                     | Provisioning Flow        | Fulfillment and Readiness Tracker                 |
| `/it/assignments`                      | Assignment Ledger        | Allocation and Return History                     |
| `/it/email`                            | Email Identities         | Active Communication Registry                     |
| `/it/email/new`                        | Email Provisioning       | Mailbox Creation Workspace                        |
| `/admin/reports`                       | Infrastructure Intelligence | Global resource overview for HR, IT, and Admin  |
| `/admin/management`                    | Master Data Management   | Govern central organizational entities           |
| `/admin/audit`                         | Audit Timeline           | System Activity and Trace Records                 |
| `/admin/users`                         | Access Control           | User Roles and Governance Dashboard               |
| *(fallback)*                           | HR Module                | Human Resources Management                        |

---

## 8. Theme System

### How It Works

1. Default theme: **dark** (set server-side by reading `mams-theme` cookie in `layout.tsx`)
2. `ThemeProvider` reads cookie on mount, applies to `document.documentElement.dataset.theme`
3. Both `data-theme="dark"` and `.dark` CSS class are toggled for Tailwind dark variant support
4. Theme persists in both `localStorage` and a cookie (1-year expiry)
5. Cookie enables SSR to pre-render correct theme without flash

### ThemeToggle Component

A pill-style toggle button:
```tsx
// Container
"inline-flex h-10 items-center gap-2 rounded-full border border-border bg-card px-3 text-sm font-semibold text-foreground shadow-sm transition-all hover:border-primary/30 hover:text-primary"

// Track
"relative flex h-6 w-11 items-center rounded-full bg-muted p-1 transition-colors"

// Thumb (sliding dot)
"absolute h-4 w-4 rounded-full bg-primary shadow-sm transition-transform"
// Dark: translate-x-5 | Light: translate-x-0
```

---

## 9. Component Design Patterns

### 9.1 Cards

```tsx
// Standard card
"bg-card border border-border/60 rounded-2xl p-4"

// Premium card (with hover lift)
"p-6 rounded-2xl premium-card glass animate-fade-in group"

// Glass card
"glass border border-border/50 rounded-2xl overflow-hidden shadow-sm"

// Large rounded card (modals, feature blocks)
"rounded-[32px]"
```

`.premium-card` CSS class:
```css
.premium-card {
  background: var(--card);
  border: 1px solid var(--border);
  box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05), 0 1px 2px 0 rgba(0,0,0,0.03);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.premium-card:hover {
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04);
  transform: translateY(-1px);
}
```

`.glass` CSS class:
```css
.glass {
  background: var(--glass-bg);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--glass-border);
}
```

### 9.2 StatsCard (`StatsCard.tsx`)

Used on the Dashboard. Props: `title`, `value`, `description`, `count`, `icon`, `trend`, `trendValue`.

```tsx
// Layout: icon top-left, trend badge top-right
// Value: text-3xl font-bold tracking-tight
// Title: text-sm font-medium text-muted-foreground uppercase tracking-widest
// Trend badge colors:
//   up:      bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400
//   down:    bg-red-100   text-red-700   dark:bg-red-900/30   dark:text-red-400
//   neutral: bg-blue-100  text-blue-700  dark:bg-blue-900/30  dark:text-blue-400

// Clickable stats card wrapping pattern:
"block rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background"

// Left accent border variants:
"border-l-4 border-l-green-500"    // Upcoming Joinings
"border-l-4 border-l-primary"      // Workforce
"border-l-4 border-l-secondary"    // Hardware
"border-l-4 border-l-accent"       // Provisioning
"border-l-4 border-l-muted-foreground" // Emails
```

### 9.3 Buttons

```tsx
// Primary action
"bg-primary text-primary-foreground hover:scale-105 active:scale-95 shadow-lg shadow-primary/20 transition-all"

// Small label button (table actions, header buttons)
"text-[9px] font-black uppercase tracking-widest rounded-xl px-4 py-1.5"

// Danger / destructive
"text-red-500 hover:bg-red-500/10 hover:text-red-500 transition-all"

// Ghost / muted
"text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg transition-all"

// Log out button
"w-full flex items-center gap-3 px-3 py-2 text-xs font-black uppercase tracking-widest text-muted-foreground hover:text-red-500 hover:bg-red-500/5 rounded-lg transition-all"
```

### 9.4 Inputs

```tsx
// Text input
"bg-muted/50 border border-border rounded-xl px-4 py-2.5 text-sm font-bold focus:bg-background outline-none transition-all"

// OmniSearch input (frameless)
"flex-1 bg-transparent py-4 text-base font-medium outline-none placeholder:text-muted-foreground/40"
```

### 9.5 Status Badges / Pills

```tsx
// Generic colored pill
"bg-{color}/10 text-{color} text-[7px] font-black uppercase px-1.5 py-0.5 rounded-full"

// Examples used in codebase:
"bg-amber-500/10 text-amber-500"   // NOTICE / Warning
"bg-red-500/10   text-red-500"     // EXIT / Danger
"bg-primary/10   text-primary"     // Active role / info
"bg-green-500/10 text-green-500"   // Active / OK

// Notification count badge
"text-[10px] font-black rounded-full bg-red-500 text-white ring-2 ring-red-500/30 animate-pulse w-5 h-5"
```

### 9.6 Section Headers (inside pages)

```tsx
// With icon + View All link pattern:
<div className="flex items-center justify-between px-1">
  <h3 className="text-sm font-black uppercase tracking-widest flex items-center gap-2">
    <Icon className="w-4 h-4 text-{color}-500" />
    Section Title
  </h3>
  <Link className="text-[10px] font-bold text-primary hover:underline">VIEW ALL</Link>
</div>
```

### 9.7 Modals / Overlays

```tsx
// Backdrop
"fixed inset-0 z-[9999] flex items-start justify-center pt-[12vh]"
"absolute inset-0 bg-background/70 backdrop-blur-xl animate-in fade-in duration-200"

// Modal card
"relative w-full max-w-2xl mx-4 animate-in fade-in slide-in-from-top-4 zoom-in-95 duration-300"
"bg-card border border-border/80 rounded-2xl shadow-[0_32px_80px_-16px_rgba(0,0,0,0.5)] overflow-hidden"
```

### 9.8 Dropdown Menus

```tsx
// Dropdown panel
"absolute right-0 mt-2 w-64 bg-card border border-border rounded-2xl shadow-2xl overflow-hidden animate-fade-in glass z-50"

// Dropdown header section
"p-4 border-b border-border bg-muted/30"

// Dropdown item
"w-full flex items-center gap-3 px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg transition-all group"
```

---

## 10. OmniSearch (`OmniSearch.tsx`)

A **Ctrl+K** command palette rendered via React Portal at `z-[9999]`.

### Trigger
- Search button in header (visible `sm:` and above, width `w-48 lg:w-64`)
- Global keyboard shortcut: `Ctrl+K` / `⌘K`
- Managed via `SearchContext` (`isOmniSearchOpen`, `openOmniSearch`, `closeOmniSearch`)

### Behavior
- Min 2 characters to search
- 300ms debounce
- AbortController cancels in-flight requests on new keystrokes
- Arrow keys to navigate results, Enter to open, Esc to close

### Category Color Map

| Category    | Color           | Badge                                        |
|-------------|-----------------|----------------------------------------------|
| employees   | `text-blue-400`   | `bg-blue-500/15 text-blue-400 border-blue-500/20`   |
| assets      | `text-emerald-400`| `bg-emerald-500/15 text-emerald-400 border-emerald-500/20` |
| accessories | `text-purple-400` | `bg-purple-500/15 text-purple-400 border-purple-500/20` |
| emails      | `text-amber-400`  | `bg-amber-500/15 text-amber-400 border-amber-500/20`  |
| workspaces  | `text-cyan-400`   | `bg-cyan-500/15 text-cyan-400 border-cyan-500/20`    |
| provisioning| `text-rose-400`   | `bg-rose-500/15 text-rose-400 border-rose-500/20`    |

---

## 11. ScopeSelector (`ScopeSelector.tsx`)

A location/company scope filter in the header.

```tsx
// Trigger button (open/closed states)
"flex items-center gap-2.5 px-4 py-2 rounded-2xl border transition-all duration-300 shadow-sm"
// Open:   "bg-primary/10 border-primary/30 ring-4 ring-primary/5"
// Closed: "bg-card/40 border-white/5 hover:bg-muted/50 hover:border-border"

// Dropdown panel
"absolute top-full left-0 mt-3 w-64 bg-card border border-border rounded-3xl shadow-2xl overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200 glass z-[60]"

// Option item (active)
"bg-primary/10 text-primary"
// Option item (inactive)
"hover:bg-muted/50 text-muted-foreground hover:text-foreground"
```

Scope is stored in cookie `x-mams-scope-location` and triggers a full page reload to apply server-side scoping.

---

## 12. Animation System

### Custom Animations

```css
/* Defined in globals.css */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
  animation: fadeIn 0.5s ease-out forwards;
}
```

### Tailwind animate-in Utilities (used in modals/dropdowns)

```tsx
"animate-in fade-in duration-200"                     // Simple fade
"animate-in fade-in slide-in-from-top-4 zoom-in-95 duration-300"  // Modal enter
"animate-in fade-in slide-in-from-top-2 duration-200"             // Dropdown
```

### Hover / Active Micro-Interactions

```tsx
// Card hover lift
"hover:scale-[1.02] transition-all"

// Button press
"hover:scale-105 active:scale-95 transition-all"

// Icon on hover
"group-hover:scale-110 transition-transform"

// Settings icon rotate
"group-hover:rotate-45 transition-transform"

// Logout icon slide
"group-hover:-translate-x-1 transition-transform"

// Chevron rotate (open state)
"transition-transform duration-200" + cn(isOpen && "rotate-180")
```

---

## 13. Icon Usage

All icons are from `lucide-react`. Sizes used:

| Size Class      | Usage                              |
|-----------------|------------------------------------|
| `w-2.5 h-2.5`  | Keyboard shortcut icons            |
| `w-3 h-3`      | ChevronDown in avatar              |
| `w-3.5 h-3.5`  | Small inline icons, search icon    |
| `w-4 h-4`      | Nav link icons, dropdown items     |
| `w-5 h-5`      | Header icons, sidebar toggle       |
| `w-7 h-7`      | OmniSearch empty state             |
| `w-10 h-10`    | Feature icon in alert/stat cards   |

---

## 14. Scrollbar Styling

```css
::-webkit-scrollbar       { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: var(--muted-foreground); }
```

Custom class `.custom-scrollbar` is applied to the sidebar's scrollable content area.

---

## 15. Provider Nesting Order

Wrap your app in exactly this order (outermost → innermost):

```tsx
<ThemeProvider>
  <AuthProvider>                 {/* NextAuth SessionProvider */}
    <SidebarProvider>            {/* isCollapsed state */}
      <ToastProvider>            {/* showToast(message, type) */}
        <NotificationProvider>   {/* polls /api/notifications every 30s */}
          <SearchProvider>       {/* OmniSearch state + Ctrl+K */}
            <SidebarWrapper>     {/* renders Sidebar + Header + main */}
              {children}
            </SidebarWrapper>
          </SearchProvider>
        </NotificationProvider>
      </ToastProvider>
    </SidebarProvider>
  </AuthProvider>
</ThemeProvider>
```

---

## 16. Dashboard Page Layout Pattern

```tsx
// Root container
<div className="space-y-12 max-w-7xl mx-auto">

  {/* Stats grid - top row */}
  <section className="animate-fade-in pt-4">
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
      {/* StatsCard per module */}
    </div>
  </section>

  {/* Content columns - lower section */}
  <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 animate-fade-in delay-100">
    {/* Column 1: Upcoming Joiners list */}
    {/* Column 2: Exits/Offboarding list */}
    {/* Column 3: Alerts / Action Required */}
  </div>
</div>
```

---

## 17. List Page Pattern (Data Tables)

Every list page (assets, employees, accessories, etc.) follows this structure:

```
┌─────────────────────────────────────────────────────────────┐
│  [FILTER BAR]                                               │
│  Search input | Dropdown filters | Export btn | Import btn  │
│  New + button (primary, right-aligned)                      │
├─────────────────────────────────────────────────────────────┤
│  [DATA TABLE]                                               │
│  bg-card rounded-2xl border border-border/60 overflow-hidden│
│  ┌──────────────────────────────────────────┐               │
│  │ Table header [bg-muted/30, uppercase,    │               │
│  │ text-[9px] font-black tracking-widest]   │               │
│  ├──────────────────────────────────────────┤               │
│  │ Row [hover:bg-muted/30, divide-y]        │               │
│  │ Row                                      │               │
│  │ Row                                      │               │
│  └──────────────────────────────────────────┘               │
│  [PAGINATION / Load more]                                   │
└─────────────────────────────────────────────────────────────┘
```

Table header cells: `text-[9px] font-black uppercase tracking-widest text-muted-foreground/60 px-4 py-3`

---

## 18. Alert / Notification Cards

Used in Dashboard "Action Required" section:

```tsx
"p-4 rounded-2xl bg-card border border-border/50 premium-card flex items-center gap-4 group flex-1 shadow-sm"

// Icon container colors by alert type:
// warning: "bg-accent/10 text-accent"
// danger:  "bg-red-500/10 text-red-500"
// info:    "bg-primary/10 text-primary"

// Alert title
"text-[10px] font-black uppercase tracking-widest text-muted-foreground/60"
// Alert description
"text-xs font-bold leading-tight"
```

---

## 19. User Avatar

Two states: photo (from disk) or initials fallback.

```tsx
// Avatar with photo
<img src={photoUrl} className="w-8 h-8 rounded-full object-cover" />

// Initials fallback
<div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold text-xs ring-2 ring-primary/10">
  {initial}
</div>

// Larger header variant (9x9)
"w-9 h-9 rounded-full bg-secondary ring-2 ring-primary ring-offset-2 ring-offset-card shadow-lg"
```

---

## 20. Quick Reference: Tailwind Class Cheatsheet

```
SPACING:   p-4, p-6, px-3, py-2, gap-3, gap-4, gap-6, space-y-4, space-y-12
ROUNDING:  rounded-lg, rounded-xl, rounded-2xl, rounded-3xl, rounded-[32px], rounded-full
BORDERS:   border border-border, border-border/50, border-border/60, divide-y divide-border/50
SHADOWS:   shadow-sm, shadow-lg, shadow-2xl, shadow-[0_32px_80px_-16px_rgba(0,0,0,0.5)]
OPACITY:   /10, /20, /30, /40, /50, /60, /80 (used on colors heavily)
BLUR:      backdrop-blur-md (header), backdrop-blur-xl (modal), backdrop-blur (12px via .glass)
TEXT SIZE: text-[7px], text-[8px], text-[9px], text-[10px], text-xs, text-sm, text-base, text-xl, text-3xl
FONT:      font-medium, font-semibold, font-bold, font-black
TRACKING:  tracking-tight, tracking-tighter, tracking-wide, tracking-widest, tracking-[0.24em], tracking-[0.2em]
TRANSFORM: scale-105, scale-[1.02], hover:scale-110, active:scale-95, -translate-x-1, rotate-180
ANIMATION: transition-all, transition-colors, transition-transform, duration-200, duration-300
Z-INDEX:   z-30 (header), z-50 (sidebar, dropdowns), z-[60] (scope selector), z-[9999] (omni search)
```

---

## 21. Adding a New Page (Checklist)

When creating a new page that follows MAMS design:

1. Add header title/subtitle entry to `UserHeader.tsx` `headerContent` array
2. Add sidebar nav link to `Sidebar.tsx` `sidebarLinks` array with correct `permission` tuple
3. Add page file at `src/app/{module}/{entity}/page.tsx`
4. Page root: `<div className="space-y-6 max-w-7xl mx-auto animate-fade-in">`
5. Section heading: `<h2 className="text-sm font-black uppercase tracking-widest">`
6. Cards: use `premium-card glass rounded-2xl p-6`
7. Tables: wrap in `<div className="bg-card rounded-2xl border border-border/60 overflow-hidden">`
8. Always check both light and dark theme rendering

---

## 22. Fonts Setup (Next.js layout.tsx)

```tsx
import { Plus_Jakarta_Sans, JetBrains_Mono } from "next/font/google";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
});

const jbMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

// Apply to body:
<body className={`${jakarta.variable} ${jbMono.variable} antialiased font-sans`}>
```

---

*End of MAMS Design System Reference*
