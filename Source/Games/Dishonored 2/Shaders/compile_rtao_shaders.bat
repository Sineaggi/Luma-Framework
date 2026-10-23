@echo off
rem Compiles the D3D12 side RTAO shaders to DXIL headers embedded in the addon (rtao_pass.cpp includes them).
rem RayQuery needs shader model 6.5, which only DXC can compile. Re-run after editing any .hlsl/.hlsli here
rem and commit the regenerated headers. Uses the DXC from the Windows SDK (dxc.exe + dxil.dll, so the output is signed).
setlocal
set "DXC="
for /f "delims=" %%d in ('dir /b /ad /o-n "%ProgramFiles(x86)%\Windows Kits\10\bin\10.*" 2^>nul') do (
   if not defined DXC if exist "%ProgramFiles(x86)%\Windows Kits\10\bin\%%d\x64\dxc.exe" set "DXC=%ProgramFiles(x86)%\Windows Kits\10\bin\%%d\x64\dxc.exe"
)
if not defined DXC (
   echo dxc.exe not found in the Windows SDK
   exit /b 1
)
echo Using "%DXC%"
pushd "%~dp0"
"%DXC%" -nologo -T cs_6_5 -E main -O3 -Fh rtao_trace_cs.h -Vn g_rtao_trace_cs rtao_trace.hlsl || goto :fail
"%DXC%" -nologo -T cs_6_5 -E main -O3 -Fh rtao_filter_cs.h -Vn g_rtao_filter_cs rtao_filter.hlsl || goto :fail
popd
echo RTAO shaders compiled
exit /b 0
:fail
popd
echo RTAO shader compilation FAILED
exit /b 1
