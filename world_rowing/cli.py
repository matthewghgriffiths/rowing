#!/usr/bin/env python

import sys
import argparse 
import datetime
from typing import (
    Optional, Dict, List, cast, Union, Any, Tuple, Iterable
)

import pandas as pd 

import cmd2

from world_rowing import api, dashboard
from world_rowing.utils import first

year_parser = argparse.ArgumentParser(
    description='Specify which year you want, defaults to current'
)
year_parser.add_argument(
    'year', type=int, help='year to retrieve', 
    nargs='?', default=datetime.datetime.now().year
    )
year_parser.add_argument(
    'choices', type=int, help='year to retrieve', 
    nargs='*', default=()
    )


n_parser = argparse.ArgumentParser(
    description='Specify how many results you want'
)
n_parser.add_argument(
    'n', type=int, help='number to retrieve', 
    nargs='?', default=5
    )


class RowingApp(cmd2.Cmd):
    """A simple cmd2 application."""
    
    def __init__(self):
        super().__init__()

        # Make maxrepeats settable at runtime
        self.current_race = api.get_last_race_started().name
        self.current_competition = api.get_most_recent_competition().name

        self.add_settable(
            cmd2.Settable('current_race', str, 'id of current race', self)
        )
        self.add_settable(
            cmd2.Settable('current_competition', int, 'id of current competition', self)
        )

        self.intro = "Welcome try running `pgmts` or `livetracker`"
        self.prompt = 'rowing> '
        self.foreground_color = 'cyan'

    def select_with_choice(
        self, 
        opts: Union[str, List[str], List[Tuple[Any, Optional[str]]]], 
        prompt: str = 'Your choice? ',
        choice: Optional[int] = None,
    ):
        if isinstance(choice, int):
            local_opts: Union[List[str], List[Tuple[Any, Optional[str]]]]
            if isinstance(opts, str):
                local_opts = cast(List[Tuple[Any, Optional[str]]], list(zip(opts.split(), opts.split())))
            else:
                local_opts = opts
            fulloptions: List[Tuple[Any, Optional[str]]] = []
            for opt in local_opts:
                if isinstance(opt, str):
                    fulloptions.append((opt, opt))
                else:
                    try:
                        fulloptions.append((opt[0], opt[1]))
                    except IndexError:
                        fulloptions.append((opt[0], opt[0]))
            try:
                option, description = fulloptions[choice - 1]
                self.poutput(f'selecting option {choice}. {description}')
                return str(option)
            except (ValueError, IndexError) as ex:
                self.poutput(
                    f"'{choice}' isn't a valid choice. "
                    f"Pick a number between 1 and {len(fulloptions)}:"
                )
                
        return self.select(opts, prompt)


    def select_from_dataframe(
            self, df, 
            column='DisplayName', name='row', 
            choice: Optional[str] = None, prompt: Optional[str] = None
    ):
        selected_id = self.select_with_choice(
            list(df[column].items()),
            prompt or f"Select which {name} you want ",
            choice=choice
        )
        return df.loc[selected_id]


    @cmd2.with_argparser(year_parser)
    def do_select_race(self, args):
        race, event, competition = self.select_race(args.year, choice=first(args.choices, None))
        self.current_race = race.name
        self.current_event = competition.name
        self.current_competition = competition.name
        self.poutput(race)

    @cmd2.with_argparser(year_parser)
    def do_pgmts(self, args):
        choices = iter(args.choices)
        competition = self.select_competition(
            args.year, choice=next(choices, None)
        )
        pgmts = api.get_competition_pgmts(competition.name)
        if pgmts.empty:
            self.poutput(
                f'no results could be loaded for {competition.DisplayName}')
            return 

        group_boat_pgmts = pgmts.groupby('Boat')
        boat_pgmts = group_boat_pgmts\
            .first()\
            .sort_values('PGMT', ascending=False)
        self.poutput(
            f"loaded PGMTS for {len(pgmts)} results"
        )
        mode = self.select_with_choice(
            [
                'by result', 
                'by boat class',
                'by final',
                'plot by boat class'
            ],
            'How to display PGMTs?',
            choice=next(choices, None)
        ) 
        if mode == 'by result':
            pgmts.PGMT = pgmts.PGMT.map("{:.2%}".format)
            self.poutput(pgmts.to_string())
        elif mode == 'by boat class':
            boat_pgmts.PGMT = boat_pgmts.PGMT.map("{:.2%}".format)
            self.poutput(boat_pgmts.to_string())
        elif mode == 'by final':
            final_pgmts = api.get_competition_pgmts(
                competition.name, finals_only=True)
            final_pgmts.PGMT = final_pgmts.PGMT.map("{:.2%}".format)
            self.poutput(final_pgmts.to_string())
        else:
            import matplotlib.pyplot as plt
            plt.ion()
            f, ax = plt.subplots(figsize=(12, 8))
            ymin = 1
            ymax = 0
            n = 10
            for boat in boat_pgmts.index:
                pgmt = group_boat_pgmts.get_group(boat).PGMT.sort_values(ascending=False)
                ax.step(
                    range(pgmt.size), pgmt.values, 
                    label=boat, where='post'
                )
                ymin = min(ymin, pgmt.values[:n].min())
                ymax = max(ymax, pgmt.values[:n].max())
                
            ax.set_xlim(0, n)
            ax.set_ylim(
                ymin - 0.01, 
                pgmts.PGMT.max() + .01)
            ax.legend()
            plt.show()

    @cmd2.with_argparser(n_parser)
    def do_upcoming(self, args):
        next_races = api.show_next_races(args.n)
        if next_races.empty:
            self.poutput(
                'Could not find any upcoming races'
            )
        else:
            self.poutput(next_races.to_string())

    @cmd2.with_argparser(year_parser)
    def do_view(self, args):
        race, event, competition = self.select_race(
            args.year, choices=args.choices)
        self.dashboard(race.name)

    def dashboard(self, race_id):
        import matplotlib.pyplot as plt
        dash = dashboard.Dashboard.from_race_id(
            race_id, 
        )
        dash.live_ion_dashboard()
        plt.show(block=False)

    def do_livetracker(self, args):
        import matplotlib.pyplot as plt
        dash = dashboard.Dashboard.load_last_race()
        dash.live_ion_dashboard()
        plt.show(block=False)



    def select_race(self, year: int, choices: Iterable[int] = ()):
        choices = iter(choices)
        competition = self.select_competition(year, next(choices, None))
        race, event = self.select_competition_race(competition.name, choices=choices)
        return race, event, competition
        
    def select_competition(self, year: int, choice: Optional[int] = None):
        return self.select_from_dataframe(
            api.get_competitions(year), name='competition', choice=choice, 
        )

    def select_competition_race(self, competition_id: str, choices: Iterable[int] = ()):
        events = api.get_competition_events(competition_id)
        if len(events) == 0:
            self.poutput("no races found for {competition.DisplayName}")
            return 

        choices = iter(choices)
        event = self.select_from_dataframe(
            events, name='event', choice=next(choices, None), 
        )
        races = api.get_competition_races(competition_id)
        if len(races) == 0:
            self.poutput("no races found for {event.DisplayName}")
            return 

        race = self.select_from_dataframe(
            races.loc[races.eventId == event.name], 
            name='race',
            choice=next(choices, None)
        )
        return race, event

    
def main():
    sys.exit(RowingApp().cmdloop())


if __name__ == '__main__':
    main()