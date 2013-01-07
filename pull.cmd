@echo off
robocopy \\lily\local\surban\dev\ml\ . /MIR /XD apps /XD .git /NJH /NJS /NDL
robocopy \\lily\local\surban\dev\ml\apps\ .\apps /PURGE /NJH /NJS /NDL
robocopy \\lily\local\surban\dev\ml\.git .\.git /PURGE /NJH /NJS /NDL /NFL

