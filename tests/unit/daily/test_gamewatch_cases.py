"""The watcher's case table: which running programs count as a game, and why the rest do not.

Each case is one record of a table row, classified under one synthetic `daily.env`. Every
folder and file name is synthetic; only the generic launcher, client, runtime and tool names
that the shipped exclusion table carries are real. A case id names its row by number and
class, never by a product.
"""

from __future__ import annotations

import pytest

from scripts.daily import gamewatch

DAILY_ENV = "\n".join(
    [
        "# The synthetic host configuration every case is classified under.",
        "MIST_RELEASE_TAG=daily-2026.11.06",
        "MIST_GAME_DIRS=X:\\Steam\\steamapps\\common;X:\\Riot;X:\\Epic Games;X:\\XboxGames;"
        "X:\\Games;X:\\Users\\player\\AppData\\Local\\ExamplePlatform\\Versions",
        "MIST_GAME_RUNTIME_DIRS=X:\\Users\\player\\AppData\\Local\\.ftba\\instances;"
        "Y:\\Packs\\Instances;X:\\Users\\player\\AppData\\Roaming\\.minecraft;Y:\\Servers",
        "MIST_GAME_EXCLUDE=X:\\Steam\\steamapps\\common\\Example Backdrop;example_tool.exe;"
        "Y:\\Servers\\Example Test Server",
        "",
    ]
)

RULES = gamewatch.rules_from_env(DAILY_ENV)

LIB = "MIST_GAME_DIRS"
RUN = "MIST_GAME_RUNTIME_DIRS"
STEAM = "X:\\Steam\\steamapps\\common"
EPIC_LAUNCHER = "X:\\Epic Games\\Launcher"
SYSTEM_JAVA = "C:\\Program Files\\Java\\jdk-21\\bin"


def record(image, cwd=None, args=None) -> gamewatch.ProcessRecord:
    return gamewatch.ProcessRecord(image=image, cwd=cwd, args=None if args is None else tuple(args))


def counts(case_id, rec, variable, index):
    return pytest.param(rec, True, (variable, index), id=case_id)


def not_counted(case_id, rec, reason):
    return pytest.param(rec, False, reason, id=case_id)


CASES = [
    counts(
        "row01-game-match-process",
        record("X:\\Riot\\Example Arena\\Game\\Example Arena.exe"),
        LIB,
        1,
    ),
    counts(
        "row02-game-with-menus-in-process-shipping",
        record(
            "X:\\Riot\\Example Tactics\\live\\ExampleTactics\\Binaries\\Win64\\"
            "ExampleTactics-Win64-Shipping.exe"
        ),
        LIB,
        1,
    ),
    counts(
        "row02-game-with-menus-in-process-starter",
        record("X:\\Riot\\Example Tactics\\live\\ExampleTactics.exe"),
        LIB,
        1,
    ),
    counts(
        "row03-game-whose-name-holds-client",
        record(
            "X:\\Riot\\Example Board\\Live\\Game\\Binaries\\Win64\\"
            "ExampleBoardClient-Win64-Shipping.exe"
        ),
        LIB,
        1,
    ),
    counts(
        "row04-store-library-game",
        record(f"{STEAM}\\Example Quest\\ExampleQuest.exe"),
        LIB,
        0,
    ),
    counts(
        "row05-dedicated-server-in-library",
        record(
            f"{STEAM}\\Example Server\\Server\\Binaries\\Win64\\ExampleServer-Win64-Shipping.exe"
        ),
        LIB,
        0,
    ),
    counts(
        "row06-second-store-library-game",
        record("X:\\Epic Games\\examplerace\\Binaries\\Win64\\ExampleRace.exe"),
        LIB,
        2,
    ),
    counts(
        "row07-app-store-library-game",
        record("X:\\XboxGames\\Example Saga\\Content\\ExampleSaga.exe"),
        LIB,
        3,
    ),
    counts(
        "row08-game-executable-beside-its-launcher",
        record("X:\\Games\\Example Agent\\Retail\\ExampleAgent.exe"),
        LIB,
        4,
    ),
    counts(
        "row09-platform-player-in-versions-folder",
        record(
            "X:\\Users\\player\\AppData\\Local\\ExamplePlatform\\Versions\\version-0a1b2c3d\\"
            "ExamplePlayerBeta.exe"
        ),
        LIB,
        5,
    ),
    counts(
        "row10-runtime-client-by-gamedir-and-cwd",
        record(
            "X:\\Users\\player\\AppData\\Local\\.ftba\\bin\\runtime\\java-runtime-delta\\bin\\"
            "javaw.exe",
            cwd="X:\\Users\\player\\AppData\\Local\\.ftba\\instances\\example pack",
            args=["--gameDir", "X:\\Users\\player\\AppData\\Local\\.ftba\\instances\\example pack"],
        ),
        RUN,
        0,
    ),
    counts(
        "row11-runtime-client-by-gamedir",
        record(
            "Y:\\Packs\\Install\\java\\Jre_21\\bin\\javaw.exe",
            args=["--gameDir", "Y:\\Packs\\Instances\\Example Pack"],
        ),
        RUN,
        1,
    ),
    counts(
        "row12-runtime-client-with-java-in-excluded-folder",
        record(
            "X:\\XboxGames\\Minecraft Launcher\\Content\\runtime\\java-runtime-gamma\\bin\\"
            "javaw.exe",
            args=["--gameDir=X:\\Users\\player\\AppData\\Roaming\\.minecraft"],
        ),
        RUN,
        2,
    ),
    counts(
        "row13-runtime-client-cwd-unreadable",
        record(
            "Y:\\Packs\\Install\\java\\Jre_21\\bin\\javaw.exe",
            cwd=None,
            args=["--gameDir", "Y:\\Packs\\Instances\\Example Pack"],
        ),
        RUN,
        1,
    ),
    counts(
        "row14-runtime-server-system-java-relative-args",
        record(
            f"{SYSTEM_JAVA}\\java.exe",
            cwd="Y:\\Servers\\Example Server",
            args=["@user_jvm_args.txt", "@libraries/example/win_args.txt", "nogui"],
        ),
        RUN,
        3,
    ),
    counts(
        "row15-runtime-server-own-java",
        record(
            "Y:\\Servers\\Example Server 2\\jre\\21\\bin\\java.exe",
            cwd="Y:\\Servers\\Example Server 2",
        ),
        RUN,
        3,
    ),
    not_counted(
        "row16-client-folder-services",
        record("X:\\Riot\\Riot Client\\RiotClientServices.exe"),
        "shipped folder riot client",
    ),
    not_counted(
        "row16-client-folder-electron",
        record("X:\\Riot\\Riot Client\\RiotClientElectron\\Riot Client.exe"),
        "shipped folder riot client",
    ),
    not_counted(
        "row16-client-folder-crash-handler",
        record("X:\\Riot\\Riot Client\\RiotClientCrashHandler.exe"),
        "shipped folder riot client",
    ),
    not_counted(
        "row17-anti-cheat-service-outside",
        record("C:\\Program Files\\Riot Vanguard\\vgc.exe"),
        "outside",
    ),
    not_counted(
        "row17-anti-cheat-tray-outside",
        record("C:\\Program Files\\Riot Vanguard\\vgtray.exe"),
        "outside",
    ),
    not_counted(
        "row18-client-lobby",
        record("X:\\Riot\\Example Arena\\LeagueClient.exe"),
        "shipped name leagueclient*.exe",
    ),
    not_counted(
        "row18-client-lobby-ux",
        record("X:\\Riot\\Example Arena\\LeagueClientUx.exe"),
        "shipped name leagueclient*.exe",
    ),
    not_counted(
        "row18-client-lobby-renderer",
        record("X:\\Riot\\Example Arena\\LeagueClientUxRender.exe"),
        "shipped name leagueclient*.exe",
    ),
    not_counted(
        "row19-store-client-outside",
        record("X:\\Steam\\steam.exe"),
        "outside",
    ),
    not_counted(
        "row19-store-web-helper-outside",
        record("X:\\Steam\\bin\\cef\\cef.win64\\steamwebhelper.exe"),
        "outside",
    ),
    not_counted(
        "row20-store-launcher",
        record(f"{EPIC_LAUNCHER}\\Portal\\Binaries\\Win64\\EpicGamesLauncher.exe"),
        "shipped folder epic games\\launcher",
    ),
    not_counted(
        "row20-store-launcher-version-selector",
        record(f"{EPIC_LAUNCHER}\\Engine\\Binaries\\Win64\\UnrealVersionSelector.exe"),
        "shipped folder epic games\\launcher",
    ),
    not_counted(
        "row20-store-launcher-overlay",
        record(f"{EPIC_LAUNCHER}\\Portal\\Extras\\Overlay\\EOSOverlayRenderer-Win64-Shipping.exe"),
        "shipped folder epic games\\launcher",
    ),
    not_counted(
        "row21-store-app-outside",
        record(
            "C:\\Program Files\\WindowsApps\\Microsoft.GamingApp_1.0.0.0_x64__8wekyb3d8bbwe\\"
            "XboxPcApp.exe"
        ),
        "outside",
    ),
    not_counted(
        "row22-launcher-folder",
        record("X:\\XboxGames\\Minecraft Launcher\\Content\\Minecraft.exe"),
        "shipped folder minecraft launcher",
    ),
    not_counted(
        "row22-launcher-folder-helper",
        record("X:\\XboxGames\\Minecraft Launcher\\Content\\gamelaunchhelper.exe"),
        "shipped folder minecraft launcher",
    ),
    not_counted(
        "row23-modpack-app-outside",
        record("X:\\Users\\player\\AppData\\Local\\Programs\\ftb-app\\FTB Electron App.exe"),
        "outside",
    ),
    not_counted(
        "row23-modpack-app-own-runtime",
        record(
            "X:\\Users\\player\\AppData\\Local\\.ftba\\runtime\\bin\\javaw.exe",
            cwd="X:\\Users\\player\\AppData\\Local\\.ftba",
            args=[],
        ),
        "no runtime folder",
    ),
    not_counted(
        "row24-modpack-app-outside",
        record("X:\\Users\\player\\AppData\\Local\\Programs\\CurseForge Windows\\CurseForge.exe"),
        "outside",
    ),
    not_counted(
        "row25-controller-tool-beside-servers",
        record("Y:\\Servers\\Example Pad Mapper\\ExamplePadMapper.exe"),
        "outside",
    ),
    not_counted(
        "row26-redistributable-shared-folder",
        record(f"{STEAM}\\Steamworks Shared\\_CommonRedist\\vcredist\\2022\\VC_redist.x64.exe"),
        "shipped folder steamworks shared",
    ),
    not_counted(
        "row26-redistributable-redist-folder",
        record(f"{STEAM}\\Example Quest\\Engine\\Extras\\Redist\\en-us\\UEPrereqSetup_x64.exe"),
        "shipped folder redist",
    ),
    not_counted(
        "row26-redistributable-directx-folder",
        record("X:\\Epic Games\\DirectXRedist\\DXSETUP.exe"),
        "shipped folder directxredist",
    ),
    not_counted(
        "row26-redistributable-installer-name",
        record(f"{STEAM}\\Example Quest\\EpicOnlineServicesInstaller.exe"),
        "shipped name *installer*.exe",
    ),
    not_counted(
        "row27-launcher-in-game-folder",
        record("X:\\Games\\Example Agent\\Launcher.exe"),
        "shipped name *launcher*.exe",
    ),
    not_counted(
        "row27-launcher-beside-game-binary",
        record("X:\\Epic Games\\examplerace\\Binaries\\Win64\\Launcher.exe"),
        "shipped name *launcher*.exe",
    ),
    not_counted(
        "row28-crash-handler",
        record(f"{STEAM}\\Example Quest\\UnityCrashHandler64.exe"),
        "shipped name *crashhandler*.exe",
    ),
    not_counted(
        "row28-crash-report-client",
        record(f"{STEAM}\\Example Quest\\Engine\\Binaries\\Win64\\CrashReportClient.exe"),
        "shipped name *crashreport*.exe",
    ),
    not_counted(
        "row28-crashpad-handler",
        record(f"{STEAM}\\Example Quest\\Plugins\\Sentry\\Binaries\\Win64\\crashpad_handler.exe"),
        "shipped name *crashpad*.exe",
    ),
    not_counted(
        "row29-anti-cheat-starter",
        record(f"{STEAM}\\Example Arena 2\\start_protected_game.exe"),
        "shipped name start_protected_game.exe",
    ),
    not_counted(
        "row29-anti-cheat-folder",
        record(f"{STEAM}\\Example Arena 2\\EasyAntiCheat\\EasyAntiCheat_EOS_Setup.exe"),
        "shipped folder easyanticheat",
    ),
    not_counted(
        "row29-anti-cheat-name",
        record("X:\\Epic Games\\examplerace\\Binaries\\Win64\\ExampleRace_EAC.exe"),
        "shipped name *_eac.exe",
    ),
    not_counted(
        "row30-vr-runtime-server",
        record(f"{STEAM}\\SteamVR\\bin\\win64\\vrserver.exe"),
        "shipped folder steamvr",
    ),
    not_counted(
        "row30-vr-runtime-compositor",
        record(f"{STEAM}\\SteamVR\\bin\\win64\\vrcompositor.exe"),
        "shipped folder steamvr",
    ),
    not_counted(
        "row30-vr-runtime-environment",
        record(f"{STEAM}\\SteamVR\\tools\\steamvr_environments\\game\\bin\\win64\\steamtours.exe"),
        "shipped folder steamvr",
    ),
    not_counted(
        "row31-background-app-host-excluded-folder",
        record(f"{STEAM}\\Example Backdrop\\backdrop64.exe"),
        "MIST_GAME_EXCLUDE[0]",
    ),
    not_counted(
        "row31-background-service-host-excluded-folder",
        record(f"{STEAM}\\Example Backdrop\\bin\\backdropservice64.exe"),
        "MIST_GAME_EXCLUDE[0]",
    ),
    not_counted(
        "row32-tool-host-excluded-name",
        record(f"{STEAM}\\Example Tool\\Example_Tool.exe"),
        "MIST_GAME_EXCLUDE[1]",
    ),
    not_counted(
        "row33-runtime-other-program",
        record(
            f"{SYSTEM_JAVA}\\java.exe",
            cwd="X:\\Users\\player\\Projects\\tool",
            args=["-jar", "tool.jar"],
        ),
        "no runtime folder",
    ),
    not_counted(
        "row34-runtime-installer-in-server-folder",
        record(
            f"{SYSTEM_JAVA}\\java.exe",
            cwd="Y:\\Servers\\Example Server",
            args=["-jar", "example-installer.jar", "--installServer"],
        ),
        "installer",
    ),
    not_counted(
        "row35-runtime-unreadable",
        record(f"{SYSTEM_JAVA}\\javaw.exe"),
        "folder unreadable",
    ),
    not_counted(
        "row36-runtime-server-in-host-excluded-folder",
        record(f"{SYSTEM_JAVA}\\java.exe", cwd="Y:\\Servers\\Example Test Server"),
        "MIST_GAME_EXCLUDE[2]",
    ),
    not_counted(
        "row37-sibling-folder-longer-name",
        record("X:\\Games2\\Example.exe"),
        "outside",
    ),
    counts(
        "row38-library-game-on-bundled-runtime",
        record(
            f"{STEAM}\\Example Spire\\jre\\bin\\java.exe",
            cwd=f"{STEAM}\\Example Spire",
            args=["-jar", "desktop-1.0.jar"],
        ),
        LIB,
        0,
    ),
]


@pytest.mark.parametrize(("rec", "expected_counts", "expected"), CASES)
def test_case_table(rec, expected_counts, expected):
    verdict = gamewatch.classify(rec, RULES)
    assert verdict.counts is expected_counts
    if expected_counts:
        assert (verdict.variable, verdict.index) == expected
    else:
        assert verdict.reason == expected
        assert (verdict.variable, verdict.index) == (None, None)


def test_every_row_of_the_table_has_a_case():
    rows = {param.id.split("-", 1)[0] for param in CASES}
    assert rows == {f"row{number:02d}" for number in range(1, 39)}


def test_the_synthetic_env_parses_to_the_three_variables():
    assert len(RULES.libraries) == 6
    assert len(RULES.runtime_dirs) == 4
    assert RULES.exclude_dirs == (
        "X:\\Steam\\steamapps\\common\\Example Backdrop",
        "",
        "Y:\\Servers\\Example Test Server",
    )
    assert RULES.exclude_names == ("", "example_tool.exe", "")


def case_record(case_id: str) -> gamewatch.ProcessRecord:
    return next(param.values[0] for param in CASES if param.id == case_id)


def test_a_counting_runtime_names_the_folder_it_counted_for():
    verdict = gamewatch.classify(case_record("row10-runtime-client-by-gamedir-and-cwd"), RULES)
    assert verdict.folder == "X:\\Users\\player\\AppData\\Local\\.ftba\\instances\\example pack"
    assert verdict.reason == "runtime"


def test_a_relative_gamedir_is_joined_to_the_working_directory():
    rec = record(f"{SYSTEM_JAVA}\\javaw.exe", cwd="Y:\\Packs", args=["--gameDir", "Instances\\a"])
    verdict = gamewatch.classify(rec, RULES)
    assert (verdict.counts, verdict.variable, verdict.index) == (True, RUN, 1)
    assert verdict.folder == "Y:\\Packs\\Instances\\a"


def test_a_relative_gamedir_without_a_working_directory_is_ignored():
    rec = record(f"{SYSTEM_JAVA}\\javaw.exe", args=["--gameDir", "Instances\\a"])
    assert gamewatch.classify(rec, RULES).reason == "no runtime folder"


def test_gamedir_is_matched_ignoring_case():
    rec = record(f"{SYSTEM_JAVA}\\javaw.exe", args=["--GAMEDIR", "y:\\packs\\instances\\a"])
    assert gamewatch.classify(rec, RULES).counts is True


@pytest.mark.parametrize("flag", ["--installServer", "-installServer", "installClient", "-INSTALL"])
def test_every_leading_dash_is_removed_before_the_installer_check(flag):
    rec = record(
        f"{SYSTEM_JAVA}\\java.exe", cwd="Y:\\Servers\\a", args=["-jar", "server.jar", flag]
    )
    assert gamewatch.classify(rec, RULES).reason == "installer"


def test_a_jar_operand_named_like_an_installer_is_an_installer():
    rec = record(
        f"{SYSTEM_JAVA}\\java.exe", cwd="Y:\\Servers\\a", args=["-jar", "libs/Mod-Installer-2.jar"]
    )
    assert gamewatch.classify(rec, RULES).reason == "installer"


def test_a_gamedir_under_a_runtime_dir_wins_over_the_working_directory():
    rec = record(
        f"{SYSTEM_JAVA}\\javaw.exe",
        cwd="Y:\\Servers\\a",
        args=["--gameDir", "Y:\\Packs\\Instances\\b"],
    )
    verdict = gamewatch.classify(rec, RULES)
    assert (verdict.index, verdict.folder) == (1, "Y:\\Packs\\Instances\\b")


def test_a_gamedir_outside_every_runtime_dir_falls_back_to_the_working_directory():
    rec = record(f"{SYSTEM_JAVA}\\javaw.exe", cwd="Y:\\Servers\\a", args=["--gameDir", "Y:\\Other"])
    verdict = gamewatch.classify(rec, RULES)
    assert (verdict.index, verdict.folder) == (3, "Y:\\Servers\\a")


def test_a_counting_library_program_names_no_folder():
    verdict = gamewatch.classify(case_record("row01-game-match-process"), RULES)
    assert verdict.folder is None
    assert verdict.reason == "library"
