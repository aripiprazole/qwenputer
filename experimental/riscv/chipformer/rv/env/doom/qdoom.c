/* Doom on the Qwenputer RV32 machine: the platform layer (I_*) and the game loop.
 *
 * Every line here and in Doom runs as guest RV32 instructions. The machine parks between tool
 * calls: a call's text (the keys to press) sits in the text buffer at TEXT_ADDR, the game runs
 * its tics, and the frame goes to the terminal as display bytes (DSP tokens, which Qwen never
 * reads). Qwen gets a one-line status in its <tool_response> through putchar.
 *
 * A frame on the display (COLOR, the default): shift-out (14), then ROWS x COLS cells of two
 * bytes, the xterm 256-color index of the cell's top and bottom pixel (the terminal draws a
 * half block), then a status line. With -DQDOOM_TEXT_FRAME: form feed (12), ROWS lines of COLS
 * brightness characters, then the status line. Every other display byte is console text
 * (Doom's startup messages, errors).
 *
 * GPL-2.0 (it links id Software's Doom source).
 */
#include "qlibc.h"

#include "doomdef.h"
#include "doomstat.h"
#include "d_event.h"
#include "d_main.h"
#include "d_net.h"
#include "g_game.h"
#include "i_system.h"
#include "i_video.h"
#include "m_argv.h"
#include "m_menu.h"
#include "r_main.h"
#include "r_draw.h"
#include "v_video.h"

#define TEXT ((const volatile unsigned char *)TEXT_ADDR)
#define ZONE_BYTES (ZONE_KB * 1024)
#define COLS 80  // the frame: the 3D view, one cell per low-detail column
#define ROWS 21  // cell rows (two pixel rows each in color, one in text)
#ifdef QDOOM_TEXT_FRAME
#define COLOR 0
#else
#define COLOR 1
#endif
#define PIXROWS (ROWS * (1 + COLOR))
#define BLOCKS 5  // view size: 160 x 84 pixels; low detail draws 80 columns
#define TICS 4    // tics a pressed key is held (35 tics a second)

extern gamestate_t wipegamestate;
extern boolean singletics, advancedemo, menuactive, automapactive;
extern int screenblocks, detailLevel;
void D_ProcessEvents(void);
void D_DoAdvanceDemo(void);
void D_Display(void);
void G_BuildTiccmd(ticcmd_t *cmd);
void R_ExecuteSetViewSize(void);

static int tics;              // the virtual clock: tics run so far
static unsigned char luma[256];  // palette index -> 0..255 brightness
static unsigned char xcol[256];  // palette index -> xterm 256-color index
static const char ramp[] = " .:-=+*#%@";

// ---------------------------------------------------------------- I_ system

void I_Init(void) {}
byte *I_ZoneBase(int *size) {
    *size = ZONE_BYTES;
    byte *zone = malloc(ZONE_BYTES);
    if (!zone) I_Error("I_ZoneBase: no %d KB for the zone", ZONE_KB);
    return zone;
}
int I_GetTime(void) { return tics; }
void I_StartFrame(void) {}
void I_StartTic(void) {}
ticcmd_t *I_BaseTiccmd(void) {
    static ticcmd_t empty;
    return &empty;
}
void I_Quit(void) { exit(0); }
byte *I_AllocLow(int length) { return calloc(1, length); }
void I_Tactile(int on, int off, int total) { (void)on; (void)off; (void)total; }
void I_Error(char *error, ...) {
    va_list ap;
    va_start(ap, error);
    printf("\nI_Error: ");
    vprintf(error, ap);
    printf("\n");
    va_end(ap);
    exit(1);
}

// ---------------------------------------------------------------- I_ video

void I_InitGraphics(void) { usegamma = 0; }
void I_ShutdownGraphics(void) {}
// xterm's color cube levels and the nearest level of an 8-bit value
static const unsigned char cube_level[6] = {0, 95, 135, 175, 215, 255};
static int cube(int v) { return v < 48 ? 0 : v < 115 ? 1 : (v - 35) / 40; }

// The palette (Doom sets it again for damage and pickup flashes): brightness for text frames,
// and the nearest xterm 256 color (the 6x6x6 cube or the 24 greys) for color frames.
void I_SetPalette(byte *palette) {
    for (int i = 0; i < 256; i++, palette += 3) {
        int r = palette[0], g = palette[1], b = palette[2];
        luma[i] = (r * 77 + g * 150 + b * 29) >> 8;
        int cr = cube(r), cg = cube(g), cb = cube(b);
        int dr = r - cube_level[cr], dg = g - cube_level[cg], db = b - cube_level[cb];
        int grey = (r + g + b) / 3, gi = grey < 8 ? 0 : grey > 238 ? 23 : (grey - 3) / 10;
        int gl = 8 + 10 * gi, er = r - gl, eg = g - gl, eb = b - gl;
        xcol[i] = dr * dr + dg * dg + db * db <= er * er + eg * eg + eb * eb
                      ? 16 + 36 * cr + 6 * cg + cb : 232 + gi;
    }
}
void I_UpdateNoBlit(void) {}
void I_FinishUpdate(void) {}
void I_WaitVBL(int count) { (void)count; }
void I_ReadScreen(byte *scr) { memcpy(scr, screens[0], SCREENWIDTH * SCREENHEIGHT); }
void I_BeginRead(void) {}
void I_EndRead(void) {}

boolean q_viewonly;  // d_main.c: only the 3D view is shown (no status bar, HUD, border)

static int view_only(void) { return gamestate == GS_LEVEL && !menuactive && !automapactive; }

// Before a frame: what it will show, and the view rows present() samples (qnextrow, r_draw.c),
// the only rows the renderer draws then. Every row when the whole screen is shown.
static void sample_rows(void) {
    static int done = -1;
    q_viewonly = view_only();
    int key = q_viewonly ? viewheight : 0;
    if (key == done) return;
    done = key;
    static unsigned char need[SCREENHEIGHT];
    memset(need, !q_viewonly, sizeof need);
    if (q_viewonly)
        for (int r = 0; r < PIXROWS; r++) need[(2 * r + 1) * viewheight / (2 * PIXROWS)] = 1;
    qnextrow[SCREENHEIGHT] = SCREENHEIGHT;
    for (int y = SCREENHEIGHT - 1; y >= 0; y--) qnextrow[y] = need[y] ? y : qnextrow[y + 1];
}

// The view window while playing; the whole screen (menus, intermission) otherwise.
static void present(void) {
    static int done_x0 = -1, done_w;
    static short colx[COLS];
    q_show(COLOR ? 14 : '\f');
    int level = view_only();
    int x0 = level ? viewwindowx : 0, y0 = level ? viewwindowy : 0;
    int w = level ? scaledviewwidth : SCREENWIDTH, h = level ? viewheight : SCREENHEIGHT;
    if (x0 != done_x0 || w != done_w) {
        for (int c = 0; c < COLS; c++) colx[c] = x0 + (2 * c + 1) * w / (2 * COLS);
        done_x0 = x0, done_w = w;
    }
#if COLOR
    for (int r = 0; r < ROWS; r++) {
        const byte *top = screens[0] + (y0 + (4 * r + 1) * h / (2 * PIXROWS)) * SCREENWIDTH;
        const byte *bot = screens[0] + (y0 + (4 * r + 3) * h / (2 * PIXROWS)) * SCREENWIDTH;
        for (int c = 0; c < COLS; c++) {
            q_show(xcol[top[colx[c]]]);
            q_show(xcol[bot[colx[c]]]);
        }
    }
#else
    for (int r = 0; r < ROWS; r++) {
        const byte *row = screens[0] + (y0 + (2 * r + 1) * h / (2 * ROWS)) * SCREENWIDTH;
        for (int c = 0; c < COLS; c++) q_show(ramp[luma[row[colx[c]]] * 10 >> 8]);
        q_show('\n');
    }
#endif
}

static const char *const weapons[] = {"fist", "pistol", "shotgun", "chaingun", "rocket launcher",
                                      "plasma rifle", "BFG 9000", "chainsaw", "super shotgun"};

// One line for Qwen (and under the frame): the player's state and the latest game message.
static void status(void (*put)(int)) {
    char line[160];
    player_t *p = &players[consoleplayer];
    if (gamestate != GS_LEVEL) {
        snprintf(line, sizeof line, "%s", gamestate == GS_INTERMISSION ? "level complete; press enter"
                 : gamestate == GS_FINALE ? "episode complete" : "title screen");
    } else {
        int ammo = weaponinfo[p->readyweapon].ammo;
        snprintf(line, sizeof line, "E%dM%d %s health %d armor %d %s ammo %d kills %d/%d%s%s%s",
                 gameepisode, gamemap, p->playerstate == PST_DEAD ? "DEAD" : "alive", p->health,
                 p->armorpoints, weapons[p->readyweapon], ammo == am_noammo ? 0 : p->ammo[ammo],
                 p->killcount, totalkills, menuactive ? " menu open" : "",
                 p->message ? " | " : "", p->message ? p->message : "");
    }
    for (const char *s = line; *s; s++) put(*s);
}

// ---------------------------------------------------------------- the tool call's keys

static char input[512];

// The call's argument: the text between the second '>' and the next '<' (the harness's rule).
static int gather(void) {
    int n = 0, gt = 0;
    for (int row = 1; row <= 80; row++) {
        const volatile unsigned char *p = TEXT + 16 * row;
        if (!p[0]) break;
        for (int k = 0; k < 16 && p[k]; k++) {
            int c = p[k];
            if (gt < 2) gt += c == '>';
            else if (c == '<') { input[n] = 0; return n; }
            else if (n < (int)sizeof input - 1) input[n++] = c;
        }
    }
    input[n] = 0;
    return n;
}

static const struct { const char *name; int key; } keys[] = {
    {"w", KEY_UPARROW}, {"up", KEY_UPARROW}, {"forward", KEY_UPARROW},
    {"s", KEY_DOWNARROW}, {"down", KEY_DOWNARROW}, {"back", KEY_DOWNARROW},
    {"a", KEY_LEFTARROW}, {"left", KEY_LEFTARROW},
    {"d", KEY_RIGHTARROW}, {"right", KEY_RIGHTARROW},
    {"q", ','}, {"strafeleft", ','}, {"e", '.'}, {"straferight", '.'},
    {"f", KEY_RCTRL}, {"fire", KEY_RCTRL}, {"shoot", KEY_RCTRL},
    {"u", ' '}, {"use", ' '}, {"open", ' '},
    {"enter", KEY_ENTER}, {"esc", KEY_ESCAPE}, {"menu", KEY_ESCAPE}, {"y", 'y'}, {"n", 'n'},
    {"tab", KEY_TAB}, {"map", KEY_TAB},
    {"1", '1'}, {"2", '2'}, {"3", '3'}, {"4", '4'}, {"5", '5'}, {"6", '6'}, {"7", '7'},
    {"wait", 0}, {"-", 0},
};

static void tic(void) {
    I_StartTic();
    D_ProcessEvents();
    G_BuildTiccmd(&netcmds[consoleplayer][maketic % BACKUPTICS]);
    if (advancedemo) D_DoAdvanceDemo();
    M_Ticker();
    G_Ticker();
    gametic++;
    maketic++;
    tics++;
}

static void post(evtype_t type, int key) {
    event_t ev = {type, key, 0, 0};
    D_PostEvent(&ev);
}

static char unknown_keys[64];

// Each word presses one key for TICS tics; "w*3" holds it three times as long. Unknown words
// are reported and skipped.
static void play(const char *p) {
    unknown_keys[0] = 0;
    char word[16];
    while (*p) {
        while (*p && isspace(*p)) p++;
        int n = 0;
        while (*p && !isspace(*p) && n < 15) word[n++] = tolower(*p++);
        word[n] = 0;
        if (!n) break;
        int hold = 1;
        char *star = strchr(word, '*');
        if (star) {
            *star = 0;
            hold = atoi(star + 1);
            if (hold < 1) hold = 1;
            if (hold > 35) hold = 35;
        }
        int key = -1;
        for (unsigned i = 0; i < sizeof keys / sizeof keys[0]; i++)
            if (!strcmp(word, keys[i].name)) key = keys[i].key;
        if (key < 0) {
            if (strlen(unknown_keys) + n + 2 < sizeof unknown_keys) strcat(strcat(unknown_keys, " "), word);
            continue;
        }
        if (key) post(ev_keydown, key);
        for (int t = 0; t < TICS * hold; t++) tic();
        if (key) post(ev_keyup, key);
    }
    tic();
}

static void respond(void) {
    q_putc('\n'); q_putc(256); q_putc('\n');  // <tool_response>
    status(q_putc);
    if (unknown_keys[0]) {
        const char *s = " | unknown_keys keys:";
        while (*s) q_putc(*s++);
        for (s = unknown_keys; *s; s++) q_putc(*s);
    }
    q_putc('\n'); q_putc(257); q_putc('\n'); q_putc(258); q_putc('\n'); q_putc(259); q_putc('\n');
}

void D_DoomLoop(void) {
    I_InitGraphics();
    singletics = true;
    R_SetViewSize(BLOCKS, 1);  // a small, low-detail view: the frame is 80 columns anyway
    R_ExecuteSetViewSize();    // now, so the first frame's sampled rows use its height
    tic();
    for (;;) {
        wipegamestate = gamestate;  // no screen melt: it would draw many frames per tic
        sample_rows();
        D_Display();
        present();
        status(q_show);
        q_show('\n');
        respond();
        q_park();
        gather();
        play(input);
    }
}

// ---------------------------------------------------------------- entry

int main(void) {
    static char *argv[] = {"doom", "-warp", "1", "1", "-skill", "3", NULL};
    myargc = 6;
    myargv = argv;
    q_park();  // boot stops here; the first tool call starts Doom
    D_DoomMain();
    return 0;
}
