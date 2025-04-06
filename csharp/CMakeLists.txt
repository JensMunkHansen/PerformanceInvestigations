cmake_minimum_required(VERSION 3.15)
project(PerformanceInvestigationsCSharp NONE)

function(add_csharp_program name cs_file)
    set(project_dir ${CMAKE_CURRENT_BINARY_DIR}/${name})
    file(MAKE_DIRECTORY ${project_dir})

    add_custom_command(
        OUTPUT ${project_dir}/bin/Release/net8.0/${name}.dll
        COMMAND dotnet new console --force --language "C#" --framework net8.0 --output ${project_dir}
        COMMAND ${CMAKE_COMMAND} -E copy ${cs_file} ${project_dir}/Program.cs
        COMMAND dotnet add ${project_dir} package BenchmarkDotNet
        COMMAND dotnet build ${project_dir} --configuration Release -p:AllowUnsafeBlocks=true
        WORKING_DIRECTORY ${project_dir}
        COMMENT "Building ${name} in Release mode"
    )

    add_custom_target(${name} ALL
        DEPENDS ${project_dir}/bin/Release/net8.0/${name}.dll
    )
endfunction()

add_csharp_program(CSBaseline ${CMAKE_CURRENT_SOURCE_DIR}/CSBaseline.cs)
add_csharp_program(CSBlockedColumnMultiOutput ${CMAKE_CURRENT_SOURCE_DIR}/CSBlockedColumnMultiOutput.cs)
add_csharp_program(CSBlockedColumnMultiOutputAVX ${CMAKE_CURRENT_SOURCE_DIR}/CSBlockedColumnMultiOutputAVX.cs)
