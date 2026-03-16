import { HistoryOutlined, PlusOutlined, SettingOutlined } from '@ant-design/icons';
import { Layout, Menu, Typography } from 'antd';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';

const { Header, Content, Sider } = Layout;

const items = [
  { key: '/tasks/new', icon: <PlusOutlined />, label: '\u65b0\u5efa\u4efb\u52a1' },
  { key: '/history', icon: <HistoryOutlined />, label: '\u5386\u53f2\u8bb0\u5f55' },
  { key: '/settings', icon: <SettingOutlined />, label: '\u8bbe\u7f6e' },
];

export function AppLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const selectedKey = location.pathname.startsWith('/tasks/') && location.pathname !== '/tasks/new' ? '/history' : location.pathname;

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider breakpoint="lg" collapsedWidth="0">
        <div className="brand">DepthSplat v3</div>
        <Menu theme="dark" mode="inline" selectedKeys={[selectedKey]} items={items} onClick={({ key }) => navigate(key)} />
      </Sider>
      <Layout>
        <Header className="app-header">
          <Typography.Title level={4} style={{ margin: 0 }}>
            {'DepthSplat \u4efb\u52a1\u63a7\u5236\u53f0'}
          </Typography.Title>
        </Header>
        <Content className="app-content">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
