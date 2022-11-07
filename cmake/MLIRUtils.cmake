function(mlir_gen_enums prefix)
    set(LLVM_TARGET_DEFINITIONS Enums.td)

    mlir_tablegen(Enums.h.inc -gen-enum-decls)
    mlir_tablegen(Enums.cpp.inc -gen-enum-defs)

    add_public_tablegen_target(${prefix}EnumsIncGen)
    add_dependencies(${prefix}IncGen ${prefix}EnumsIncGen)
endfunction()

function(mlir_gen_iface prefix iface kind)
    set(LLVM_TARGET_DEFINITIONS ${iface}.td)

    mlir_tablegen(${iface}.h.inc -gen-${kind}-interface-decls)
    mlir_tablegen(${iface}.cpp.inc -gen-${kind}-interface-defs)

    add_public_tablegen_target(${prefix}${iface}InterfaceIncGen)
    add_dependencies(${prefix}IncGen ${prefix}${iface}InterfaceIncGen)

    add_mlir_doc(${iface} ${iface} Interfaces/ -gen-${kind}-interface-docs)
endfunction()