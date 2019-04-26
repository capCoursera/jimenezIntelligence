#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from configargparse import ArgumentParser
from mechanicalsoup import StatefulBrowser

from SMACB.SuperManager import SuperManagerACB
from SMACB.TemporadaACB import TemporadaACB

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add('-u', dest='user', type=str, env_var='SM_USER', required=True)
    parser.add('-p', dest='password', type=str, env_var='SM_PASSWORD', required=True)

    parser.add('-v', dest='verbose', action="count", env_var='SM_VERBOSE', required=False, default=0)
    parser.add('-d', dest='debug', action="store_true", env_var='SM_DEBUG', required=False, default=False)

    parser.add('-i', dest='infile', type=str, env_var='SM_INFILE', required=True)
    parser.add('-o', dest='outfile', type=str, env_var='SM_OUTFILE', required=False)
    parser.add('-t', dest='temporada', type=str, env_var='SM_TEMPORADA', required=True)
    parser.add('-l', dest='league', type=str, env_var='SM_LEAGUE', required=False, default=None)

    parser.add('-j', dest='jornada', type=str, env_var='SM_JORNADA', required=False, default=None)

    args = parser.parse_args()

    browser = StatefulBrowser(soup_config={'features': "html.parser"}, raise_on_404=True, user_agent="SMparser", )
    if 'verbose' in args:
        browser.set_verbose(args.verbose)

    if 'debug' in args:
        browser.set_debug(args.debug)

    sm = SuperManagerACB(ligaPrivada=args.league)
    sm.loadData(args.infile)

    sm.Connect(browser=browser, config=args)

    sm.getSMstatus(browser=browser, config=args)

    if sm.changed and ('outfile' in args) and args.outfile:
        print("There were changes!")
        sm.saveData(args.outfile)
