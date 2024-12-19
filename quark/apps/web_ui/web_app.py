import argparse
import datetime
import os
import pathlib
import signal
from threading import Thread, Event
from urllib.parse import urljoin

from algo_engine.apps.bokeh_server import DocManager, DocServer
from algo_engine.profile import Profile, PROFILE

from .. import LOGGER


class WebApp(object):
    def __init__(self, start_date: datetime.date, end_date: datetime.date, name: str = 'WebApp.Dashboard', profile: Profile = None, **kwargs):
        from flask import Flask

        self.start_date = start_date
        self.end_date = end_date
        self.name = name
        self.root_dir = pathlib.Path(__file__).parent
        self.profile = PROFILE if profile is None else profile
        self.host = kwargs.get('host', '0.0.0.0')
        self.port = kwargs.get('port', 8080)
        self.stop_event = Event()
        self.server_thread = None

        self.flask = Flask(
            import_name=self.name,
            template_folder=self.root_dir.joinpath('templates'),
            static_folder=self.root_dir.joinpath('static'),
            root_path=str(self.root_dir)
        )
        self.doc_manager = DocManager(
            host='localhost' if self.host == '0.0.0.0' else self.host,
            port=self.port,
            bokeh_host=kwargs.get('bokeh_host', 'localhost' if self.host == '0.0.0.0' else self.host),
            bokeh_port=kwargs.get('bokeh_port', 5006),
        )
        self.contents: dict[str, dict[str, DocServer]] = {}

    def update(self, **kwargs):
        for doc_server in self.doc_manager.doc_server.values():
            with doc_server.lock:
                doc_server.update(**kwargs)

    def register(self, ticker: str, **kwargs):
        if ticker in self.contents:
            raise ValueError(f'Ticker {ticker} already registered.')

        from .docserver import Dashboard, StateBanner

        contents = self.contents[ticker] = {}
        # _candlestick = dashboard[f'candlesticks'] = Candlestick(ticker=ticker, start_date=self.start_date, end_date=self.end_date, **kwargs)
        _dashboard = contents[f'dashboard'] = Dashboard(ticker=ticker, start_date=self.start_date, end_date=self.end_date, **kwargs)
        _state_banner = contents[f'banner'] = StateBanner(ticker=ticker, start_date=self.start_date, end_date=self.end_date, **kwargs)

        # self.doc_manager.register(url=f'/candlesticks/{ticker}', doc_server=_candlestick)
        self.doc_manager.register(url=f'/dashboard/{ticker}', doc_server=_dashboard)
        self.doc_manager.register(url=f'/banner/{ticker}', doc_server=_state_banner)

    def render_index(self):
        from flask import render_template

        contents_url = {ticker: urljoin(self.url, ticker) for ticker in self.contents}

        html = render_template(
            'index.html',
            title=f'PyAlgoEngine.Backtest.App',
            data=contents_url
        )

        return html

    def render_contents(self, ticker: str):
        from flask import render_template
        from bokeh.embed import server_document

        contents = self.contents[ticker]
        bokeh_scripts = {}

        for name, doc_server in contents.items():
            url = self.doc_manager.doc_url[doc_server]
            doc_script = server_document(url=f'http://0.0.0.0:{self.doc_manager.bokeh_port}{url}')
            doc_script = doc_script.replace('0.0.0.0', self.doc_manager.bokeh_host)
            bokeh_scripts[name] = doc_script

        html = render_template(
            'dash.html',
            ticker=ticker,
            framework="flask",
            **bokeh_scripts
        )
        return html

    def serve(self, blocking: bool = True):
        from waitress import serve

        LOGGER.info(f'starting {self} service...')

        self.doc_manager.start()
        self.flask.route(rule='/', methods=["GET"])(self.render_index)

        for ticker in self.contents:
            def renderer():
                return self.render_contents(ticker=ticker)

            self.flask.route(rule=f'/{ticker}', methods=["GET"])(renderer)

        if blocking:
            return serve(app=self.flask, host=self.host, port=self.port)

        self.server_thread = Thread(target=self._run_server, kwargs=dict(app=self.flask, host=self.host, port=self.port))
        self.server_thread.start()

        # Monkey patch to resolve flask double logging issues
        for hdl in (logger := self.flask.logger).handlers:
            logger.removeHandler(hdl)

        for hdl in (logger := LOGGER.root).handlers:
            logger.removeHandler(hdl)

    def _run_server(self, app, host, port):
        from waitress import serve

        # Save the current server's process ID
        # self.server_pid = os.getpid()

        try:
            # Start the server and keep running until it is stopped
            serve(app=app, host=host, port=port)
        except Exception as e:
            LOGGER.error(f"Error in server: {e}")
        finally:
            LOGGER.info("Server has stopped.")

    def stop(self, timeout=1):
        LOGGER.info(f'Stopping {self} service...')
        # Signal the server to stop
        self.stop_event.set()

        # Perform any other cleanup, like stopping doc_manager
        self.doc_manager.stop()

        # Check if the server is running in a thread
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=timeout)  # Wait for the thread to finish execution

        # If the server PID is set, kill the process to release the port
        if hasattr(self, 'server_pid'):
            os.kill(self.server_pid, signal.SIGTERM)  # Graceful termination
            self.server_pid = None

    @property
    def url(self) -> str:
        if self.host == '0.0.0.0':
            return f'http://localhost:{self.port}/'
        else:
            return f'http://{self.host}:{self.port}/'


def start_app(start_date: datetime.date, end_date: datetime.date, blocking: bool = True, **kwargs):
    web_app = WebApp(start_date=start_date, end_date=end_date, **kwargs)
    web_app.serve(blocking=blocking)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Backtest.App')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')

    args = parser.parse_args()

    start_app(
        start_date=datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date(),
        end_date=datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date(),
    )
