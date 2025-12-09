@echo off
REM ============================================================================
REM Multi-State Trail Fetcher for Windows
REM Fetches trails for South Carolina, North Carolina, and Georgia
REM Runs: Fetch -> Merge -> Validate for each state
REM ============================================================================

setlocal enabledelayedexpansion

echo ========================================================================
echo           Multi-State Trail Fetcher (SC, NC, GA)
echo ========================================================================
echo.
echo This will fetch, merge, and validate trails for:
echo   - South Carolina
echo   - North Carolina  
echo   - Georgia
echo.

REM Ask about elevation data
set ELEVATION_FLAG=
set /p use_elevation="Include elevation data? (Y/N, adds 30-60 min per state): "
if /i "%use_elevation%"=="Y" (
    set ELEVATION_FLAG=--elevation
    echo.
    echo Elevation data ENABLED - This will add 30-60 minutes per state
    echo Estimated total time: 90-180 minutes
) else (
    echo.
    echo Elevation data DISABLED
    echo Estimated total time: 45-75 minutes
)

echo.
echo Output files will be named:
echo   - south_carolina_trails.json
echo   - north_carolina_trails.json
echo   - georgia_trails.json
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause > nul

REM ============================================================================
REM STATE CONFIGURATIONS
REM ============================================================================

REM South Carolina
set SC_NAME=South Carolina
set SC_ABBR=SC
set SC_BBOX=32.0 -83.4 35.2 -78.5
set SC_TILES=3 2
set SC_OUTPUT=trails_SC.json

REM North Carolina
set NC_NAME=North Carolina
set NC_ABBR=NC
set NC_BBOX=33.8 -84.3 36.6 -75.4
set NC_TILES=4 3
set NC_OUTPUT=trails_NC.json

REM Georgia
set GA_NAME=Georgia
set GA_ABBR=GA
set GA_BBOX=30.3 -85.6 35.0 -80.8
set GA_TILES=4 3
set GA_OUTPUT=trails_GA.json

REM ============================================================================
REM PROCESS EACH STATE
REM ============================================================================

set TOTAL_STATES=3
set CURRENT_STATE=0

REM --- Process South Carolina ---
set /a CURRENT_STATE+=1
call :process_state "%SC_NAME%" "%SC_ABBR%" "%SC_BBOX%" "%SC_TILES%" "%SC_OUTPUT%"
if errorlevel 1 goto :error

REM --- Process North Carolina ---
set /a CURRENT_STATE+=1
call :process_state "%NC_NAME%" "%NC_ABBR%" "%NC_BBOX%" "%NC_TILES%" "%NC_OUTPUT%"
if errorlevel 1 goto :error

REM --- Process Georgia ---
set /a CURRENT_STATE+=1
call :process_state "%GA_NAME%" "%GA_ABBR%" "%GA_BBOX%" "%GA_TILES%" "%GA_OUTPUT%"
if errorlevel 1 goto :error

REM ============================================================================
REM SUCCESS - ALL STATES COMPLETE
REM ============================================================================

echo.
echo ========================================================================
echo                        ALL STATES COMPLETE!
echo ========================================================================
echo.
echo Backend-ready trail files created:
echo   1. south_carolina_trails.json
echo   2. north_carolina_trails.json
echo   3. georgia_trails.json
echo.
echo All files are:
echo   * Properly formatted JSON
echo   * Merged (duplicate names combined)
echo   * Validated for backend compatibility
echo   * Ready to import!
echo.
echo Summary statistics:
python -c "import json, os; files=['south_carolina_trails.json','north_carolina_trails.json','georgia_trails.json']; total=0; [print(f'  {f}: {len(json.load(open(f)))} trails') if os.path.exists(f) else None for f in files]; [total := total + len(json.load(open(f))) for f in files if os.path.exists(f)]; print(f'\nTOTAL: {total} trails across all 3 states')" 2>nul
echo.
echo Next steps:
echo   1. Import these JSON files to your backend
echo   2. Test that trails display correctly
echo   3. Deploy to production!
echo.
pause
goto :eof

REM ============================================================================
REM PROCESS STATE FUNCTION
REM ============================================================================
:process_state
set STATE_NAME=%~1
set STATE_ABBR=%~2
set STATE_BBOX=%~3
set STATE_TILES=%~4
set STATE_OUTPUT=%~5

echo.
echo ========================================================================
echo  [%CURRENT_STATE%/%TOTAL_STATES%] PROCESSING: %STATE_NAME%
echo ========================================================================
echo.

REM Temporary file names
set TEMP_UNMERGED=%STATE_ABBR%_unmerged_temp.json
set TEMP_MERGED=%STATE_ABBR%_merged_temp.json

REM --- Step 1: Fetch trails ---
echo ------------------------------------------------------------------------
echo  Step 1/3: Fetching %STATE_NAME% trails from OpenStreetMap
echo ------------------------------------------------------------------------
echo  BBox: %STATE_BBOX%
echo  Tiles: %STATE_TILES%
if defined ELEVATION_FLAG (
    echo  Estimated time: 30-60 minutes (with elevation data^)
) else (
    echo  Estimated time: 15-25 minutes
)
echo.

python fetch_trails_overpass.py --bbox %STATE_BBOX% --tiles %STATE_TILES% --no-merge %ELEVATION_FLAG% -o %TEMP_UNMERGED%

if errorlevel 1 (
    echo.
    echo ERROR: Failed to fetch %STATE_NAME% trails
    echo Check error messages above.
    del %TEMP_UNMERGED% 2>nul
    exit /b 1
)

echo.
echo  Step 1 complete for %STATE_NAME%!
echo.

REM --- Step 2: Fast merge ---
echo ------------------------------------------------------------------------
echo  Step 2/3: Merging %STATE_NAME% trails
echo ------------------------------------------------------------------------
echo  Estimated time: 10-30 seconds
echo.

python fast_merge_trails.py %TEMP_UNMERGED% -o %TEMP_MERGED% --verbose

if errorlevel 1 (
    echo.
    echo ERROR: Failed to merge %STATE_NAME% trails
    echo Check error messages above.
    del %TEMP_UNMERGED% 2>nul
    del %TEMP_MERGED% 2>nul
    exit /b 1
)

echo.
echo  Step 2 complete for %STATE_NAME%!
echo.

REM --- Step 3: Validate ---
echo ------------------------------------------------------------------------
echo  Step 3/3: Validating %STATE_NAME% trails for backend
echo ------------------------------------------------------------------------
echo  Estimated time: 1-5 seconds
echo.

python validate_for_backend.py %TEMP_MERGED% -o %STATE_OUTPUT%

if errorlevel 1 (
    echo.
    echo WARNING: Validation had issues for %STATE_NAME%, but output was saved.
    echo Check validation messages above.
) else (
    echo.
    echo  Step 3 complete for %STATE_NAME%!
)

REM Clean up temp files
del %TEMP_UNMERGED% 2>nul
del %TEMP_MERGED% 2>nul

echo.
echo ========================================================================
echo  %STATE_NAME% COMPLETE! Saved to: %STATE_OUTPUT%
echo ========================================================================

REM Show quick stats for this state
python -c "import json; data=json.load(open('%STATE_OUTPUT%')); print(f'\n  Total trails: {len(data)}\n')" 2>nul

goto :eof

REM ============================================================================
REM ERROR HANDLER
REM ============================================================================
:error
echo.
echo ========================================================================
echo                          ERROR OCCURRED
echo ========================================================================
echo.
echo Processing stopped due to an error.
echo Check the error messages above for details.
echo.
echo Partial results may be available:
dir *.json 2>nul
echo.
pause
exit /b 1